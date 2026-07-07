"""OpenAI-compatible provider for all non-Anthropic LLM APIs."""

from __future__ import annotations

import asyncio
import collections
import hashlib
import json
import os
import re
import secrets
import string
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

import json_repair
from loguru import logger
from openai import AsyncOpenAI

from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest

if TYPE_CHECKING:
    from nanobot.providers.registry import ProviderSpec

_ALLOWED_MSG_KEYS = frozenset({
    "role", "content", "tool_calls", "tool_call_id", "name",
    "reasoning_content", "extra_content",
})
_ALNUM = string.ascii_letters + string.digits

_STANDARD_TC_KEYS = frozenset({"id", "type", "index", "function"})
_STANDARD_FN_KEYS = frozenset({"name", "arguments"})
_DEFAULT_OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://github.com/HKUDS/nanobot",
    "X-OpenRouter-Title": "nanobot",
    "X-OpenRouter-Categories": "cli-agent,personal-agent",
}

# Model families that use the "reasoning" request shape: they require
# `max_completion_tokens` instead of `max_tokens` and reject custom
# `temperature` values. Detection is by model id (post prefix-strip) so it
# applies regardless of which provider hosts the model (OpenAI direct,
# GitHub Copilot, Azure, OpenRouter, etc.).
_REASONING_MODEL_PREFIXES: tuple[str, ...] = (
    "o1", "o3", "o4",
    "gpt-5",
)


def _is_reasoning_model(model_name: str) -> bool:
    """True if the model uses the OpenAI "reasoning" request shape."""
    if not model_name:
        return False
    bare = model_name.split("/")[-1].lower()
    for prefix in _REASONING_MODEL_PREFIXES:
        if bare == prefix or bare.startswith(prefix + "-") or bare.startswith(prefix + "."):
            return True
    return False


def _short_tool_id() -> str:
    """9-char alphanumeric ID compatible with all providers (incl. Mistral)."""
    return "".join(secrets.choice(_ALNUM) for _ in range(9))


def _get(obj: Any, key: str) -> Any:
    """Get a value from dict or object attribute, returning None if absent."""
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _coerce_dict(value: Any) -> dict[str, Any] | None:
    """Try to coerce *value* to a dict; return None if not possible or empty."""
    if value is None:
        return None
    if isinstance(value, dict):
        return value if value else None
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, dict) and dumped:
            return dumped
    return None


def _extract_tc_extras(tc: Any) -> tuple[
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
]:
    """Extract (extra_content, provider_specific_fields, fn_provider_specific_fields).

    Works for both SDK objects and dicts.  Captures Gemini ``extra_content``
    verbatim and any non-standard keys on the tool-call / function.
    """
    extra_content = _coerce_dict(_get(tc, "extra_content"))

    tc_dict = _coerce_dict(tc)
    prov = None
    fn_prov = None
    if tc_dict is not None:
        leftover = {k: v for k, v in tc_dict.items()
                    if k not in _STANDARD_TC_KEYS and k != "extra_content" and v is not None}
        if leftover:
            prov = leftover
        fn = _coerce_dict(tc_dict.get("function"))
        if fn is not None:
            fn_leftover = {k: v for k, v in fn.items()
                          if k not in _STANDARD_FN_KEYS and v is not None}
            if fn_leftover:
                fn_prov = fn_leftover
    else:
        prov = _coerce_dict(_get(tc, "provider_specific_fields"))
        fn_obj = _get(tc, "function")
        if fn_obj is not None:
            fn_prov = _coerce_dict(_get(fn_obj, "provider_specific_fields"))

    return extra_content, prov, fn_prov


def _uses_openrouter_attribution(spec: "ProviderSpec | None", api_base: str | None) -> bool:
    """Apply Nanobot attribution headers to OpenRouter requests by default."""
    if spec and spec.name == "openrouter":
        return True
    return bool(api_base and "openrouter" in api_base.lower())


class OpenAICompatProvider(LLMProvider):
    """Unified provider for all OpenAI-compatible APIs.

    Receives a resolved ``ProviderSpec`` from the caller — no internal
    registry lookups needed.
    """

    _QUOTA_KEY_COOLDOWN_S = 30 * 60.0  # 30m — re-probe exhausted keys within the same session
    _RATE_LIMIT_KEY_COOLDOWN_S = 60.0
    _OVERLOAD_KEY_COOLDOWN_S = 15.0

    def __init__(
        self,
        api_key: str | None = None,
        api_keys: list[str] | None = None,
        api_base: str | None = None,
        default_model: str = "gpt-4o",
        extra_headers: dict[str, str] | None = None,
        rate_limit: int = 0,
        timeout: float = 60.0,
        spec: ProviderSpec | None = None,
        token_provider: Callable[[], str] | None = None,
    ):
        key_pool = [k for k in (api_keys or []) if k]
        if not key_pool and api_key:
            key_pool = [api_key]
        super().__init__(key_pool[0] if key_pool else api_key, api_base)
        self.default_model = default_model
        self.extra_headers = extra_headers or {}
        self._spec = spec
        self._token_provider = token_provider
        self._api_keys = key_pool
        self._request_start_index = 0
        self._preferred_key_index: int | None = 0 if key_pool else None
        self._key_retry_after: list[float] = [0.0] * len(key_pool)
        self._rate_limit = max(0, int(rate_limit))
        self._request_timeout = max(5.0, float(timeout))
        self._request_timestamps: collections.deque[float] = collections.deque()

        effective_base = api_base or (spec.default_api_base if spec else None) or None
        self._is_opencode_go = bool(spec and spec.name == "opencode_go")
        self._opencode_session_id = uuid.uuid4().hex
        if self._is_opencode_go:
            default_headers = {
                "x-opencode-session": self._opencode_session_id,
                "x-opencode-client": "nanobot",
                "User-Agent": "nanobot",
            }
        else:
            default_headers = {"x-session-affinity": self._opencode_session_id}
        if _uses_openrouter_attribution(spec, effective_base):
            default_headers.update(_DEFAULT_OPENROUTER_HEADERS)
        if extra_headers:
            default_headers.update(extra_headers)
        self._effective_base = effective_base
        self._default_headers = default_headers

        if self.api_key and self._spec and self._spec.env_key:
            self._setup_env(self.api_key, self.api_base)

    def _key_label(self, index: int | None = None) -> str:
        """Return a safe label for the active or requested key slot."""
        if not self._api_keys:
            return "key#1/1"
        idx = 0 if index is None else (index % len(self._api_keys))
        digest = hashlib.sha1(self._api_keys[idx].encode()).hexdigest()[:8]
        return f"key#{idx + 1}/{len(self._api_keys)}[{digest}]"

    def _next_request_start(self) -> int:
        """Round-robin the starting key so concurrent requests do not share mutable state."""
        if len(self._api_keys) <= 1:
            return 0
        idx = self._request_start_index % len(self._api_keys)
        self._request_start_index = (idx + 1) % len(self._api_keys)
        return idx

    def _rotate_candidate_indices(self, indices: list[int]) -> list[int]:
        """Rotate a candidate key list for fair fallback ordering."""
        if len(indices) <= 1:
            return list(indices)
        offset = self._next_request_start() % len(indices)
        return indices[offset:] + indices[:offset]

    def _next_request_order(self) -> list[int]:
        """Choose key order for a request, preferring healthy recent winners."""
        if not self._api_keys:
            return []
        if len(self._api_keys) == 1:
            return [0]

        now = time.monotonic()
        ready = [idx for idx, retry_after in enumerate(self._key_retry_after) if retry_after <= now]
        cooling = [idx for idx, retry_after in enumerate(self._key_retry_after) if retry_after > now]
        preferred = self._preferred_key_index

        order: list[int] = []
        if preferred is not None and preferred in ready:
            order.append(preferred)
            ready.remove(preferred)

        order.extend(self._rotate_candidate_indices(ready))
        cooling.sort(key=lambda idx: (self._key_retry_after[idx], idx))
        order.extend(cooling)

        return order or list(range(len(self._api_keys)))

    @staticmethod
    def _error_status_code(error: Exception) -> int | None:
        """Extract an HTTP-like status code from SDK exceptions when available."""
        status = getattr(error, "status_code", None)
        if status is None:
            resp = getattr(error, "response", None)
            status = getattr(resp, "status_code", None) if resp is not None else None
        return status if isinstance(status, int) else None

    def _cooldown_seconds_for_error(self, error: Exception) -> float:
        """Return a cooldown for keys that just failed with provider throttling/quota."""
        if isinstance(error, TimeoutError):
            return self._OVERLOAD_KEY_COOLDOWN_S
        status = self._error_status_code(error)
        summary = self._error_summary(error).lower()
        if status == 503 or any(marker in summary for marker in (
            "high demand",
            "service unavailable",
            "temporarily unavailable",
            "unavailable",
            "timeout",
            "timed out",
        )):
            return self._OVERLOAD_KEY_COOLDOWN_S
        if self._is_quota_exhaustion(summary) or any(marker in summary for marker in (
            "current quota",
            "plan and billing",
            "resource_exhausted",
        )):
            return self._QUOTA_KEY_COOLDOWN_S
        return self._RATE_LIMIT_KEY_COOLDOWN_S

    def _record_key_success(self, key_index: int | None) -> None:
        """Promote a successful key to the preferred starting slot."""
        if key_index is None or not self._api_keys:
            return
        resolved_index = key_index % len(self._api_keys)
        self._preferred_key_index = resolved_index
        self._key_retry_after[resolved_index] = 0.0

    def _record_key_failure(self, key_index: int | None, error: Exception) -> float:
        """Temporarily deprioritize a key that just failed with throttling/quota."""
        if key_index is None or not self._api_keys:
            return 0.0
        resolved_index = key_index % len(self._api_keys)
        cooldown_s = self._cooldown_seconds_for_error(error)
        self._key_retry_after[resolved_index] = max(
            self._key_retry_after[resolved_index],
            time.monotonic() + cooldown_s,
        )
        return cooldown_s

    def _build_client(self, key_index: int | None = None) -> AsyncOpenAI:
        """Build a client for the given key slot without mutating provider state."""
        active_key = self.api_key
        if self._api_keys:
            resolved_index = 0 if key_index is None else (key_index % len(self._api_keys))
            active_key = self._api_keys[resolved_index]
        if self._token_provider is not None:
            try:
                active_key = self._token_provider()
            except Exception as exc:
                raise RuntimeError(f"Failed to obtain auth token: {exc}") from exc
        if active_key and self._spec and self._spec.env_key:
            self._setup_env(active_key, self.api_base)
        return AsyncOpenAI(
            api_key=active_key or "no-key",
            base_url=self._effective_base,
            default_headers=self._default_headers,
            timeout=self._request_timeout,
        )

    async def _await_with_timeout(self, awaitable: Awaitable[Any], *, phase: str) -> Any:
        """Bound a provider SDK awaitable so a bad upstream call cannot hang forever."""
        try:
            return await asyncio.wait_for(awaitable, timeout=self._request_timeout)
        except asyncio.TimeoutError as exc:
            raise TimeoutError(f"provider {phase} timed out after {self._request_timeout:.0f}s") from exc

    async def _collect_stream_chunks(
        self,
        stream: Any,
        on_content_delta: Callable[[str], Awaitable[None]] | None,
    ) -> list[Any]:
        """Consume a streaming response with an idle timeout between chunks."""
        chunks: list[Any] = []
        iterator = stream.__aiter__()
        try:
            while True:
                try:
                    chunk = await asyncio.wait_for(anext(iterator), timeout=self._request_timeout)
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError as exc:
                    raise TimeoutError(
                        f"provider stream timed out after {self._request_timeout:.0f}s waiting for the next chunk"
                    ) from exc

                chunks.append(chunk)
                if on_content_delta and getattr(chunk, "choices", None):
                    text = getattr(chunk.choices[0].delta, "content", None)
                    if text:
                        await on_content_delta(text)
        finally:
            close = getattr(stream, "aclose", None)
            if callable(close):
                try:
                    await close()
                except Exception:
                    logger.debug("Ignoring stream close failure")
        return chunks

    def _log_rotation(self, previous_index: int, next_index: int, error: Exception, cooldown_s: float) -> None:
        """Log a request-local rotation between configured key slots."""
        logger.warning(
            "Rotating provider API key from {} to {} after quota/rate-limit response (cooldown {:.0f}s): {}",
            self._key_label(previous_index),
            self._key_label(next_index),
            cooldown_s,
            self._error_summary(error),
        )

    @staticmethod
    def _is_rate_limit_error(exc: Exception) -> bool:
        """Return True when exception indicates quota/rate limiting or temporary overload."""
        status = getattr(exc, "status_code", None)
        if status is None:
            resp = getattr(exc, "response", None)
            status = getattr(resp, "status_code", None) if resp is not None else None
        if status in (429, 503):
            return True
        msg = str(exc).lower()
        return any(m in msg for m in (
            "rate limit",
            "rate_limit",
            "quota",
            "resource_exhausted",
            "429",
            "timeout",
            "timed out",
            "unavailable",
            "service unavailable",
            "high demand",
            "503",
        ))

    async def _wait_for_rate_limit(self) -> None:
        """Sleep as needed to honor per-provider requests/minute rate_limit."""
        if self._rate_limit <= 0:
            return

        now = time.monotonic()
        window_s = 60.0
        while self._request_timestamps and self._request_timestamps[0] <= now - window_s:
            self._request_timestamps.popleft()

        if len(self._request_timestamps) >= self._rate_limit:
            oldest = self._request_timestamps[0]
            delay = oldest + window_s - now + 1.0
            if delay > 0:
                await asyncio.sleep(delay)

        self._request_timestamps.append(time.monotonic())

    def _setup_env(self, api_key: str, api_base: str | None) -> None:
        """Set environment variables based on provider spec."""
        spec = self._spec
        if not spec or not spec.env_key:
            return
        # Keep env vars aligned with the currently active key so follow-up
        # requests and auxiliary SDK behavior do not get stuck on the first key.
        os.environ[spec.env_key] = api_key
        effective_base = api_base or spec.default_api_base
        for env_name, env_val in spec.env_extras:
            resolved = env_val.replace("{api_key}", api_key).replace("{api_base}", effective_base)
            os.environ[env_name] = resolved

    @staticmethod
    def _error_summary(error: Exception) -> str:
        """Return a short provider error summary suitable for logs/messages."""
        body = getattr(error, "doc", None) or getattr(getattr(error, "response", None), "text", None)
        text = body.strip() if isinstance(body, str) and body.strip() else str(error).strip()
        return text[:240]

    @staticmethod
    def _apply_cache_control(
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
        """Inject cache_control markers for prompt caching."""
        cache_marker = {"type": "ephemeral"}
        new_messages = list(messages)

        def _mark(msg: dict[str, Any]) -> dict[str, Any]:
            content = msg.get("content")
            if isinstance(content, str):
                return {**msg, "content": [
                    {"type": "text", "text": content, "cache_control": cache_marker},
                ]}
            if isinstance(content, list) and content:
                nc = list(content)
                nc[-1] = {**nc[-1], "cache_control": cache_marker}
                return {**msg, "content": nc}
            return msg

        if new_messages and new_messages[0].get("role") == "system":
            new_messages[0] = _mark(new_messages[0])
        if len(new_messages) >= 3:
            new_messages[-2] = _mark(new_messages[-2])

        new_tools = tools
        if tools:
            new_tools = list(tools)
            new_tools[-1] = {**new_tools[-1], "cache_control": cache_marker}
        return new_messages, new_tools

    @staticmethod
    def _normalize_tool_call_id(tool_call_id: Any) -> Any:
        """Normalize to a provider-safe 9-char alphanumeric form."""
        if not isinstance(tool_call_id, str):
            return tool_call_id
        if len(tool_call_id) == 9 and tool_call_id.isalnum():
            return tool_call_id
        return hashlib.sha1(tool_call_id.encode()).hexdigest()[:9]

    def _sanitize_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Strip non-standard keys, normalize tool_call IDs."""
        sanitized = LLMProvider._sanitize_request_messages(messages, _ALLOWED_MSG_KEYS)
        id_map: dict[str, str] = {}

        def map_id(value: Any) -> Any:
            if not isinstance(value, str):
                return value
            return id_map.setdefault(value, self._normalize_tool_call_id(value))

        for clean in sanitized:
            if isinstance(clean.get("tool_calls"), list):
                normalized = []
                for tc in clean["tool_calls"]:
                    if not isinstance(tc, dict):
                        normalized.append(tc)
                        continue
                    tc_clean = dict(tc)
                    tc_clean["id"] = map_id(tc_clean.get("id"))
                    normalized.append(tc_clean)
                clean["tool_calls"] = normalized
            if "tool_call_id" in clean and clean["tool_call_id"]:
                clean["tool_call_id"] = map_id(clean["tool_call_id"])
        return sanitized

    # ------------------------------------------------------------------
    # Build kwargs
    # ------------------------------------------------------------------

    def _build_kwargs(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        model: str | None,
        max_tokens: int,
        temperature: float | None,
        reasoning_effort: str | None,
        tool_choice: str | dict[str, Any] | None,
    ) -> dict[str, Any]:
        model_name = model or self.default_model
        spec = self._spec

        if spec and spec.supports_prompt_caching:
            messages, tools = self._apply_cache_control(messages, tools)

        if spec and spec.strip_model_prefix:
            model_name = model_name.split("/")[-1]

        if spec and spec.name == "opencode_go" and model_name.startswith("minimax-"):
            raise ValueError(
                "OpenCode Go MiniMax models use the Anthropic-style /messages endpoint and "
                "are not yet supported by Nanobot's OpenAI-compatible provider. "
                "Please use another OpenCode Go model for now."
            )

        messages = self._split_tool_images_from_messages(messages)
        kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": self._sanitize_messages(self._sanitize_empty_content(messages)),
        }
        if temperature is not None:
            kwargs["temperature"] = temperature

        reasoning_model = _is_reasoning_model(model_name)
        if reasoning_model:
            # Reasoning models (o1/o3/o4/gpt-5 family) reject `max_tokens` and
            # custom `temperature`; they require `max_completion_tokens` and
            # only accept the default temperature.
            kwargs["max_completion_tokens"] = max(1, max_tokens)
            kwargs.pop("temperature", None)
        elif spec and getattr(spec, "supports_max_completion_tokens", False):
            kwargs["max_completion_tokens"] = max(1, max_tokens)
        else:
            kwargs["max_tokens"] = max(1, max_tokens)

        if spec:
            model_lower = model_name.lower()
            for pattern, overrides in spec.model_overrides:
                if pattern in model_lower:
                    kwargs.update(overrides)
                    break

        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice or "auto"

        if self._is_opencode_go:
            kwargs["extra_headers"] = {
                "x-opencode-request": uuid.uuid4().hex,
            }
            # MiMo and DeepSeek models don't support custom temperature; match
            # opencode TUI behavior (transform.ts::temperature()).
            if not any(p in model_lower for p in ("glm", "kimi", "qwen", "minimax")):
                kwargs.pop("temperature", None)

        return kwargs

    # ------------------------------------------------------------------
    # Responses API (/v1/responses)
    # ------------------------------------------------------------------
    #
    # Some providers (notably GitHub Copilot) require the Responses API for
    # reasoning models when function tools are attached — chat/completions
    # rejects `reasoning_effort` in that combination with:
    #   "Function tools with reasoning_effort are not supported for gpt-5.x
    #    in /v1/chat/completions. Please use /v1/responses instead."
    # We route there only for reasoning-model requests on opted-in specs so
    # the existing chat/completions code path keeps handling everything else.

    def _should_use_responses_api(self, model_name: str) -> bool:
        spec = self._spec
        if not spec or not getattr(spec, "use_responses_api", False):
            return False
        return _is_reasoning_model(model_name)

    @staticmethod
    def _stringify_tool_output(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    # Skip image blocks — they're carried separately as
                    # synthetic user input_image items (see _split_tool_images).
                    if item.get("type") in ("image_url", "input_image"):
                        continue
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                        continue
                    parts.append(json.dumps(item, ensure_ascii=False))
                elif isinstance(item, str):
                    parts.append(item)
                else:
                    parts.append(str(item))
            return "".join(parts)

    @staticmethod
    def _extract_image_data_url(block: Any) -> str | None:
        """Return the data URL string from a chat-completions image_url block."""
        if not isinstance(block, dict):
            return None
        if block.get("type") != "image_url":
            return None
        iu = block.get("image_url")
        if isinstance(iu, str):
            return iu
        if isinstance(iu, dict):
            url = iu.get("url")
            if isinstance(url, str):
                return url
        return None

    @classmethod
    def _split_tool_images(cls, content: Any) -> list[str]:
        """Return any image data URLs embedded in a tool message's content."""
        if not isinstance(content, list):
            return []
        urls: list[str] = []
        for item in content:
            url = cls._extract_image_data_url(item)
            if url:
                urls.append(url)
        return urls

    @classmethod
    def _split_tool_images_from_messages(
        cls,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Split image_url blocks out of tool messages into user messages.

        Some providers (e.g. Xiaomi) reject image_url blocks inside
        role=tool messages.  This mirrors what the Responses API path already
        does (see ``_messages_to_responses_input``), keeping the chat
        completions path compatible with the same providers.
        """
        result: list[dict[str, Any]] = []
        for msg in messages:
            if msg.get("role") != "tool":
                result.append(msg)
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                result.append(msg)
                continue
            image_urls = cls._split_tool_images(content)
            if not image_urls:
                result.append(msg)
                continue
            text_only: list[Any] = [
                item for item in content
                if not (isinstance(item, dict) and item.get("type") == "image_url")
            ]
            tool_msg = dict(msg)
            tool_msg["content"] = text_only if text_only else "(tool returned image(s))"
            result.append(tool_msg)
            user_parts: list[Any] = [
                {"type": "text",
                 "text": f"[tool {msg.get('name') or ''} returned the following image(s)]".strip()},
            ]
            for url in image_urls:
                user_parts.append({"type": "image_url", "image_url": {"url": url}})
            result.append({"role": "user", "content": user_parts})
        return result

    # ------------------------------------------------------------------
    # Vision fallback: describe images via a vision-capable model when
    # the primary model rejects image content.
    # ------------------------------------------------------------------

    async def _describe_images_via_fallback(
        self,
        messages: list[dict[str, Any]],
        *,
        reason: str = "vision",
    ) -> list[dict[str, Any]] | None:
        """Describe images via the configured vision fallback model.

        Collects all ``image_url`` blocks from *messages*, sends them to the
        vision fallback model (e.g. ``mimo-v2.5``) with a "describe this
        image" prompt, and replaces each ``image_url`` block with a ``text``
        block containing the description.

        Returns the modified messages, or ``None`` if no fallback model is
        configured or the fallback call failed (caller falls back to
        :meth:`_strip_image_content`).
        """
        spec = self._spec
        if not spec or not getattr(spec, "vision_fallback_model", None):
            return None

        # Collect image_url blocks and their (msg_idx, blk_idx) locations.
        image_blocks: list[dict[str, Any]] = []
        image_locations: set[tuple[int, int]] = set()
        for msg_idx, msg in enumerate(messages):
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for blk_idx, block in enumerate(content):
                if isinstance(block, dict) and block.get("type") == "image_url":
                    image_blocks.append(block)
                    image_locations.add((msg_idx, blk_idx))

        if not image_blocks:
            return None

        vision_model = spec.vision_fallback_model
        # Build a single user message asking the vision model to describe
        # every image, preserving order so we can map descriptions back.
        desc_parts: list[dict[str, Any]] = [{
            "type": "text",
            "text": (
                "Describe each image below. For each image, provide a concise "
                "but complete description including: UI elements, visible text, "
                "layout, colors, and current state. Number your descriptions "
                "Image 1, Image 2, etc. in the order they appear."
            ),
        }]
        desc_parts.extend(image_blocks)
        vision_messages: list[dict[str, Any]] = [
            {"role": "user", "content": desc_parts},
        ]

        try:
            response = await self._safe_chat(
                messages=vision_messages,
                tools=None,
                model=vision_model,
                max_tokens=2048,
                temperature=0.3,
                reasoning_effort=None,
                tool_choice=None,
            )
        except Exception as exc:
            logger.warning(
                "Vision fallback (model={}) call failed: {}. "
                "Falling back to image stripping.",
                vision_model, exc,
            )
            return None

        if response.finish_reason == "error" or not response.content:
            logger.warning(
                "Vision fallback (model={}) returned an error: {}. "
                "Falling back to image stripping.",
                vision_model, (response.content or "")[:200],
            )
            return None

        description = response.content.strip()
        # Split by "Image N" markers if the model followed instructions.
        segments = re.split(r"\n(?=Image \d+)", description)
        if len(segments) >= len(image_blocks):
            descriptions = [s.strip() for s in segments[:len(image_blocks)]]
        else:
            # Model didn't number them; use the whole description for each image.
            descriptions = [description] * len(image_blocks)

        # Replace image_url blocks with text descriptions.
        result: list[dict[str, Any]] = []
        img_counter = 0
        for msg_idx, msg in enumerate(messages):
            content = msg.get("content")
            if not isinstance(content, list):
                result.append(msg)
                continue
            new_content: list[Any] = []
            for blk_idx, block in enumerate(content):
                if (msg_idx, blk_idx) in image_locations:
                    path = (block.get("_meta") or {}).get("path", "")
                    desc = descriptions[img_counter] if img_counter < len(descriptions) else description
                    note = f"[Image described by {vision_model}"
                    if path:
                        note += f" (path={path})"
                    note += f"]:\n{desc}"
                    new_content.append({"type": "text", "text": note})
                    img_counter += 1
                else:
                    new_content.append(block)
            result.append({**msg, "content": new_content})
        return result

    @classmethod
    def _content_to_responses_parts(
        cls,
        content: Any,
        role: str,
    ) -> list[dict[str, Any]] | str | None:
        """Translate chat-completions content (str/list) to Responses API parts.

        - text blocks become input_text (user/tool) or output_text (assistant).
        - image_url blocks become input_image with a string data URL (the
          Responses API rejects the object form `{url: ...}`).
        """
        if content is None:
            return None
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return cls._stringify_tool_output(content)
        text_type = "output_text" if role == "assistant" else "input_text"
        out: list[dict[str, Any]] = []
        for item in content:
            if isinstance(item, dict):
                t = item.get("type")
                if t == "image_url":
                    url = cls._extract_image_data_url(item)
                    if url:
                        # Responses API: image_url must be a plain string.
                        out.append({"type": "input_image", "image_url": url})
                    continue
                if t in ("text", "input_text", "output_text"):
                    txt = item.get("text")
                    if isinstance(txt, str):
                        out.append({"type": text_type, "text": txt})
                    continue
                # Pass through already-Responses-shaped parts unchanged.
                out.append(item)
            elif isinstance(item, str):
                out.append({"type": text_type, "text": item})
        return out or None

    @staticmethod
    def _stringify_arguments(arguments: Any) -> str:
        if arguments is None:
            return "{}"
        if isinstance(arguments, str):
            return arguments
        try:
            return json.dumps(arguments, ensure_ascii=False)
        except TypeError:
            return str(arguments)

    @classmethod
    def _messages_to_responses_input(
        cls,
        messages: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Translate chat-completions messages to Responses API input items.

        Returns (input_items, instructions). System messages are collapsed
        into the top-level `instructions` field; tool/assistant tool_calls
        become typed function_call / function_call_output items.
        """
        input_items: list[dict[str, Any]] = []
        instruction_parts: list[str] = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            if role == "system":
                text = cls._stringify_tool_output(content)
                if text:
                    instruction_parts.append(text)
                continue
            if role == "tool":
                call_id = msg.get("tool_call_id") or ""
                input_items.append({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": cls._stringify_tool_output(content),
                })
                # The Responses API has no field for image output on
                # function_call_output, so any image content returned by a
                # tool must be re-attached as a subsequent user input_image.
                image_urls = cls._split_tool_images(content)
                if image_urls:
                    parts: list[dict[str, Any]] = [
                        {"type": "input_text",
                         "text": f"[tool {msg.get('name') or ''} returned the following image(s)]".strip()},
                    ]
                    for url in image_urls:
                        parts.append({"type": "input_image", "image_url": url})
                    input_items.append({"role": "user", "content": parts})
                continue
            if role == "assistant":
                converted = cls._content_to_responses_parts(content, role)
                if isinstance(converted, str) and converted:
                    input_items.append({
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": converted}],
                    })
                elif isinstance(converted, list) and converted:
                    input_items.append({"role": "assistant", "content": converted})
                tool_calls = msg.get("tool_calls") or []
                for tc in tool_calls:
                    if not isinstance(tc, dict):
                        continue
                    fn = tc.get("function") or {}
                    input_items.append({
                        "type": "function_call",
                        "call_id": tc.get("id") or "",
                        "name": fn.get("name") or "",
                        "arguments": cls._stringify_arguments(fn.get("arguments")),
                    })
                continue
            # user / developer / other
            converted = cls._content_to_responses_parts(content, role or "user")
            if converted is None:
                continue
            input_items.append({"role": role or "user", "content": converted})
        instructions = "\n\n".join(p for p in instruction_parts if p) or None
        return input_items, instructions

    @staticmethod
    def _tools_to_responses(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        """Flatten chat-completions function tools to Responses API shape."""
        if not tools:
            return None
        out: list[dict[str, Any]] = []
        for t in tools:
            if not isinstance(t, dict):
                out.append(t)
                continue
            if t.get("type") == "function" and isinstance(t.get("function"), dict):
                fn = t["function"]
                flat: dict[str, Any] = {
                    "type": "function",
                    "name": fn.get("name") or "",
                }
                if fn.get("description") is not None:
                    flat["description"] = fn["description"]
                params = fn.get("parameters")
                if params is not None:
                    flat["parameters"] = params
                if fn.get("strict") is not None:
                    flat["strict"] = fn["strict"]
                out.append(flat)
            else:
                # Built-in / non-function tools (web_search, file_search, etc.)
                # already use the flat Responses shape.
                out.append(t)
        return out or None

    @staticmethod
    def _tool_choice_to_responses(
        tool_choice: str | dict[str, Any] | None,
    ) -> str | dict[str, Any] | None:
        if tool_choice is None:
            return None
        if isinstance(tool_choice, str):
            return tool_choice
        if isinstance(tool_choice, dict):
            if tool_choice.get("type") == "function" and isinstance(tool_choice.get("function"), dict):
                return {
                    "type": "function",
                    "name": tool_choice["function"].get("name") or "",
                }
        return tool_choice

    def _build_responses_kwargs(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        model: str | None,
        max_tokens: int,
        reasoning_effort: str | None,
        tool_choice: str | dict[str, Any] | None,
    ) -> dict[str, Any]:
        model_name = model or self.default_model
        spec = self._spec
        if spec and spec.strip_model_prefix:
            model_name = model_name.split("/")[-1]

        sanitized = self._sanitize_messages(self._sanitize_empty_content(messages))
        input_items, instructions = self._messages_to_responses_input(sanitized)

        kwargs: dict[str, Any] = {
            "model": model_name,
            "input": input_items,
            "max_output_tokens": max(1, max_tokens),
        }
        if instructions:
            kwargs["instructions"] = instructions
        if reasoning_effort:
            kwargs["reasoning"] = {"effort": reasoning_effort}

        flat_tools = self._tools_to_responses(tools)
        if flat_tools:
            kwargs["tools"] = flat_tools
            kwargs["tool_choice"] = self._tool_choice_to_responses(tool_choice) or "auto"

        return kwargs

    def _parse_responses(self, response: Any) -> LLMResponse:
        response_map = self._maybe_mapping(response)
        output_items: list[Any]
        if response_map is not None:
            output_items = response_map.get("output") or []
            usage_obj = response_map.get("usage")
            status = response_map.get("status") or response_map.get("incomplete_details") or ""
        else:
            output_items = list(getattr(response, "output", None) or [])
            usage_obj = getattr(response, "usage", None)
            status = getattr(response, "status", "") or ""

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[ToolCallRequest] = []

        for item in output_items:
            item_map = self._maybe_mapping(item) or {}
            item_type = item_map.get("type")
            if item_type == "message":
                for c in item_map.get("content") or []:
                    c_map = self._maybe_mapping(c) or {}
                    c_type = c_map.get("type")
                    if c_type in ("output_text", "text"):
                        text = c_map.get("text")
                        if isinstance(text, str):
                            content_parts.append(text)
                    elif c_type == "refusal":
                        refusal = c_map.get("refusal")
                        if isinstance(refusal, str):
                            content_parts.append(refusal)
            elif item_type == "function_call":
                raw_args = item_map.get("arguments")
                if isinstance(raw_args, str):
                    try:
                        args = json_repair.loads(raw_args) if raw_args else {}
                    except Exception:
                        args = {}
                elif isinstance(raw_args, dict):
                    args = raw_args
                else:
                    args = {}
                call_id = item_map.get("call_id") or item_map.get("id") or _short_tool_id()
                tool_calls.append(ToolCallRequest(
                    id=str(call_id),
                    name=str(item_map.get("name") or ""),
                    arguments=args if isinstance(args, dict) else {},
                ))
            elif item_type == "reasoning":
                for s in item_map.get("summary") or []:
                    s_map = self._maybe_mapping(s) or {}
                    if s_map.get("type") == "summary_text":
                        text = s_map.get("text")
                        if isinstance(text, str):
                            reasoning_parts.append(text)

        usage_map = self._maybe_mapping(usage_obj) or {}
        if usage_map:
            prompt_tokens = int(
                usage_map.get("input_tokens")
                or usage_map.get("prompt_tokens")
                or 0
            )
            completion_tokens = int(
                usage_map.get("output_tokens")
                or usage_map.get("completion_tokens")
                or 0
            )
            total_tokens = int(
                usage_map.get("total_tokens")
                or (prompt_tokens + completion_tokens)
            )
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }
        else:
            usage = {}

        if tool_calls:
            finish_reason = "tool_calls"
        elif status == "incomplete":
            finish_reason = "length"
        else:
            finish_reason = "stop"

        return LLMResponse(
            content="".join(content_parts) or None,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            reasoning_content="\n".join(reasoning_parts) or None,
        )

    async def _collect_responses_stream(
        self,
        stream: Any,
        on_content_delta: Callable[[str], Awaitable[None]] | None,
    ) -> Any:
        """Consume a Responses API event stream; return the final response object."""
        final_response: Any = None
        iterator = stream.__aiter__()
        try:
            while True:
                try:
                    event = await asyncio.wait_for(anext(iterator), timeout=self._request_timeout)
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError as exc:
                    raise TimeoutError(
                        f"provider stream timed out after {self._request_timeout:.0f}s waiting for the next chunk"
                    ) from exc

                ev_type = getattr(event, "type", None) or (
                    event.get("type") if isinstance(event, dict) else None
                )
                if ev_type == "response.output_text.delta" and on_content_delta:
                    delta = getattr(event, "delta", None)
                    if delta is None and isinstance(event, dict):
                        delta = event.get("delta")
                    if isinstance(delta, str) and delta:
                        await on_content_delta(delta)
                elif ev_type in ("response.completed", "response.incomplete"):
                    final_response = getattr(event, "response", None)
                    if final_response is None and isinstance(event, dict):
                        final_response = event.get("response")
                elif ev_type == "response.failed":
                    err_obj = getattr(event, "response", None) or (
                        event.get("response") if isinstance(event, dict) else None
                    )
                    err_map = self._maybe_mapping(err_obj) or {}
                    error_info = err_map.get("error") or {}
                    msg = (
                        (self._maybe_mapping(error_info) or {}).get("message")
                        if isinstance(error_info, dict)
                        else None
                    ) or str(error_info) or "Responses API request failed"
                    raise RuntimeError(msg)
        finally:
            close = getattr(stream, "aclose", None)
            if callable(close):
                try:
                    await close()
                except Exception:
                    logger.debug("Ignoring stream close failure")
        return final_response

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _maybe_mapping(value: Any) -> dict[str, Any] | None:
        if isinstance(value, dict):
            return value
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            dumped = model_dump()
            if isinstance(dumped, dict):
                return dumped
        return None

    @classmethod
    def _extract_text_content(cls, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts: list[str] = []
            for item in value:
                item_map = cls._maybe_mapping(item)
                if item_map:
                    text = item_map.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                        continue
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    parts.append(text)
                    continue
                if isinstance(item, str):
                    parts.append(item)
            return "".join(parts) or None
        return str(value)

    @classmethod
    def _extract_usage(cls, response: Any) -> dict[str, int]:
        usage_obj = None
        response_map = cls._maybe_mapping(response)
        if response_map is not None:
            usage_obj = response_map.get("usage")
        elif hasattr(response, "usage") and response.usage:
            usage_obj = response.usage

        usage_map = cls._maybe_mapping(usage_obj)
        if usage_map is not None:
            return {
                "prompt_tokens": int(usage_map.get("prompt_tokens") or 0),
                "completion_tokens": int(usage_map.get("completion_tokens") or 0),
                "total_tokens": int(usage_map.get("total_tokens") or 0),
            }

        if usage_obj:
            return {
                "prompt_tokens": getattr(usage_obj, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(usage_obj, "completion_tokens", 0) or 0,
                "total_tokens": getattr(usage_obj, "total_tokens", 0) or 0,
            }
        return {}

    def _parse(self, response: Any) -> LLMResponse:
        if isinstance(response, str):
            return LLMResponse(content=response, finish_reason="stop")

        response_map = self._maybe_mapping(response)
        if response_map is not None:
            choices = response_map.get("choices") or []
            if not choices:
                content = self._extract_text_content(
                    response_map.get("content") or response_map.get("output_text")
                )
                if content is not None:
                    return LLMResponse(
                        content=content,
                        finish_reason=str(response_map.get("finish_reason") or "stop"),
                        usage=self._extract_usage(response_map),
                    )
                return LLMResponse(content="Error: API returned empty choices.", finish_reason="error")

            choice0 = self._maybe_mapping(choices[0]) or {}
            msg0 = self._maybe_mapping(choice0.get("message")) or {}
            content = self._extract_text_content(msg0.get("content"))
            finish_reason = str(choice0.get("finish_reason") or "stop")

            raw_tool_calls: list[Any] = []
            reasoning_content = msg0.get("reasoning_content")
            for ch in choices:
                ch_map = self._maybe_mapping(ch) or {}
                m = self._maybe_mapping(ch_map.get("message")) or {}
                tool_calls = m.get("tool_calls")
                if isinstance(tool_calls, list) and tool_calls:
                    raw_tool_calls.extend(tool_calls)
                    if ch_map.get("finish_reason") in ("tool_calls", "stop"):
                        finish_reason = str(ch_map["finish_reason"])
                if not content:
                    content = self._extract_text_content(m.get("content"))
                if not reasoning_content:
                    reasoning_content = m.get("reasoning_content")

            parsed_tool_calls = []
            for tc in raw_tool_calls:
                tc_map = self._maybe_mapping(tc) or {}
                fn = self._maybe_mapping(tc_map.get("function")) or {}
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    args = json_repair.loads(args)
                ec, prov, fn_prov = _extract_tc_extras(tc)
                parsed_tool_calls.append(ToolCallRequest(
                    id=_short_tool_id(),
                    name=str(fn.get("name") or ""),
                    arguments=args if isinstance(args, dict) else {},
                    extra_content=ec,
                    provider_specific_fields=prov,
                    function_provider_specific_fields=fn_prov,
                ))

            return LLMResponse(
                content=content,
                tool_calls=parsed_tool_calls,
                finish_reason=finish_reason,
                usage=self._extract_usage(response_map),
                reasoning_content=reasoning_content if isinstance(reasoning_content, str) else None,
            )

        if not response.choices:
            return LLMResponse(content="Error: API returned empty choices.", finish_reason="error")

        choice = response.choices[0]
        msg = choice.message
        content = msg.content
        finish_reason = choice.finish_reason

        raw_tool_calls: list[Any] = []
        for ch in response.choices:
            m = ch.message
            if hasattr(m, "tool_calls") and m.tool_calls:
                raw_tool_calls.extend(m.tool_calls)
                if ch.finish_reason in ("tool_calls", "stop"):
                    finish_reason = ch.finish_reason
            if not content and m.content:
                content = m.content

        tool_calls = []
        for tc in raw_tool_calls:
            args = tc.function.arguments
            if isinstance(args, str):
                args = json_repair.loads(args)
            ec, prov, fn_prov = _extract_tc_extras(tc)
            tool_calls.append(ToolCallRequest(
                id=_short_tool_id(),
                name=tc.function.name,
                arguments=args,
                extra_content=ec,
                provider_specific_fields=prov,
                function_provider_specific_fields=fn_prov,
            ))

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason or "stop",
            usage=self._extract_usage(response),
            reasoning_content=(
                getattr(msg, "reasoning_content", None)
                or (getattr(msg, "model_extra", None) or {}).get("reasoning_content")
                or None
            ),
        )

    @classmethod
    def _parse_chunks(cls, chunks: list[Any]) -> LLMResponse:
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tc_bufs: dict[int, dict[str, Any]] = {}
        finish_reason = "stop"
        usage: dict[str, int] = {}

        def _accum_tc(tc: Any, idx_hint: int) -> None:
            """Accumulate one streaming tool-call delta into *tc_bufs*."""
            tc_index: int = _get(tc, "index") if _get(tc, "index") is not None else idx_hint
            buf = tc_bufs.setdefault(tc_index, {
                "id": "", "name": "", "arguments": "",
                "extra_content": None, "prov": None, "fn_prov": None,
            })
            tc_id = _get(tc, "id")
            if tc_id:
                buf["id"] = str(tc_id)
            fn = _get(tc, "function")
            if fn is not None:
                fn_name = _get(fn, "name")
                if fn_name:
                    buf["name"] = str(fn_name)
                fn_args = _get(fn, "arguments")
                if fn_args:
                    buf["arguments"] += str(fn_args)
            ec, prov, fn_prov = _extract_tc_extras(tc)
            if ec:
                buf["extra_content"] = ec
            if prov:
                buf["prov"] = prov
            if fn_prov:
                buf["fn_prov"] = fn_prov

        for chunk in chunks:
            if isinstance(chunk, str):
                content_parts.append(chunk)
                continue

            chunk_map = cls._maybe_mapping(chunk)
            if chunk_map is not None:
                choices = chunk_map.get("choices") or []
                if not choices:
                    usage = cls._extract_usage(chunk_map) or usage
                    text = cls._extract_text_content(
                        chunk_map.get("content") or chunk_map.get("output_text")
                    )
                    if text:
                        content_parts.append(text)
                    continue
                choice = cls._maybe_mapping(choices[0]) or {}
                if choice.get("finish_reason"):
                    finish_reason = str(choice["finish_reason"])
                delta = cls._maybe_mapping(choice.get("delta")) or {}
                text = cls._extract_text_content(delta.get("content"))
                if text:
                    content_parts.append(text)
                rc = delta.get("reasoning_content")
                if isinstance(rc, str) and rc:
                    reasoning_parts.append(rc)
                for idx, tc in enumerate(delta.get("tool_calls") or []):
                    _accum_tc(tc, idx)
                usage = cls._extract_usage(chunk_map) or usage
                continue

            if not chunk.choices:
                usage = cls._extract_usage(chunk) or usage
                continue
            choice = chunk.choices[0]
            if choice.finish_reason:
                finish_reason = choice.finish_reason
            delta = choice.delta
            if delta and delta.content:
                content_parts.append(delta.content)
            if delta:
                rc = getattr(delta, "reasoning_content", None)
                if isinstance(rc, str) and rc:
                    reasoning_parts.append(rc)
            for tc in (delta.tool_calls or []) if delta else []:
                _accum_tc(tc, getattr(tc, "index", 0))

        return LLMResponse(
            content="".join(content_parts) or None,
            tool_calls=[
                ToolCallRequest(
                    id=b["id"] or _short_tool_id(),
                    name=b["name"],
                    arguments=json_repair.loads(b["arguments"]) if b["arguments"] else {},
                    extra_content=b.get("extra_content"),
                    provider_specific_fields=b.get("prov"),
                    function_provider_specific_fields=b.get("fn_prov"),
                )
                for b in tc_bufs.values()
            ],
            finish_reason=finish_reason,
            usage=usage,
            reasoning_content="".join(reasoning_parts) or None,
        )

    @staticmethod
    def _handle_error(e: Exception) -> LLMResponse:
        body = getattr(e, "doc", None) or getattr(getattr(e, "response", None), "text", None)
        msg = f"Error: {body.strip()[:500]}" if body and body.strip() else f"Error calling LLM: {e}"
        return LLMResponse(content=msg, finish_reason="error")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        max_attempts = max(1, len(self._api_keys))
        key_order = self._next_request_order()
        attempted_labels: list[str] = []
        last_error: Exception | None = None
        for attempt in range(max_attempts):
            key_index = key_order[attempt] if self._api_keys else None
            attempted_labels.append(self._key_label(key_index))
            try:
                resolved_model = (model or self.default_model) or ""
                use_responses = self._should_use_responses_api(resolved_model)
                if use_responses:
                    kwargs = self._build_responses_kwargs(
                        messages, tools, model, max_tokens,
                        reasoning_effort, tool_choice,
                    )
                else:
                    kwargs = self._build_kwargs(
                        messages, tools, model, max_tokens, temperature,
                        reasoning_effort, tool_choice,
                    )
                await self._wait_for_rate_limit()
                if len(self._api_keys) > 1:
                    logger.info(
                        "Provider request attempt {}/{} using {} model={}",
                        attempt + 1,
                        max_attempts,
                        self._key_label(key_index),
                        kwargs["model"],
                    )
                client = self._build_client(key_index)
                if use_responses:
                    raw = await self._await_with_timeout(
                        client.responses.create(**kwargs),
                        phase="request",
                    )
                    response = self._parse_responses(raw)
                else:
                    raw = await self._await_with_timeout(
                        client.chat.completions.create(**kwargs),
                        phase="request",
                    )
                    response = self._parse(raw)
                self._record_key_success(key_index)
                if attempt > 0 and self._api_keys:
                    logger.info(
                        "Provider request recovered on {} after {} attempt(s)",
                        self._key_label(key_index),
                        attempt + 1,
                    )
                return response
            except Exception as e:
                last_error = e
                has_remaining_keys = attempt + 1 < max_attempts
                if has_remaining_keys and self._is_rate_limit_error(e) and self._api_keys:
                    cooldown_s = self._record_key_failure(key_index, e)
                    self._log_rotation(key_index or 0, key_order[attempt + 1], e, cooldown_s)
                    continue
                if self._is_rate_limit_error(e) and self._api_keys:
                    self._record_key_failure(key_index, e)
                    break
                return self._handle_error(e)
        if last_error and self._is_rate_limit_error(last_error) and self._api_keys:
            logger.error(
                "All configured API keys were rate-limited or out of quota after trying {} keys ({}); last error: {}",
                len(self._api_keys),
                ", ".join(attempted_labels),
                self._error_summary(last_error),
            )
            return LLMResponse(
                content=(
                    "Error: All configured API keys were rate-limited or out of quota "
                    f"after trying {len(self._api_keys)} keys. Last error: {self._error_summary(last_error)}"
                ),
                finish_reason="error",
            )
        return self._handle_error(last_error or RuntimeError("All API keys exhausted"))

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        on_content_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> LLMResponse:
        max_attempts = max(1, len(self._api_keys))
        key_order = self._next_request_order()
        attempted_labels: list[str] = []
        last_error: Exception | None = None

        delivered_any = False
        effective_delta = on_content_delta
        if on_content_delta is not None:
            user_delta = on_content_delta

            async def _tracked_delta(text: str) -> None:
                nonlocal delivered_any
                delivered_any = True
                await user_delta(text)

            effective_delta = _tracked_delta

        for attempt in range(max_attempts):
            key_index = key_order[attempt] if self._api_keys else None
            attempted_labels.append(self._key_label(key_index))
            try:
                resolved_model = (model or self.default_model) or ""
                use_responses = self._should_use_responses_api(resolved_model)
                if use_responses:
                    kwargs = self._build_responses_kwargs(
                        messages, tools, model, max_tokens,
                        reasoning_effort, tool_choice,
                    )
                else:
                    kwargs = self._build_kwargs(
                        messages, tools, model, max_tokens, temperature,
                        reasoning_effort, tool_choice,
                    )
                kwargs["stream"] = True
                if not use_responses:
                    kwargs["stream_options"] = {"include_usage": True}
                await self._wait_for_rate_limit()
                if len(self._api_keys) > 1:
                    logger.info(
                        "Provider stream attempt {}/{} using {} model={}",
                        attempt + 1,
                        max_attempts,
                        self._key_label(key_index),
                        kwargs["model"],
                    )
                client = self._build_client(key_index)
                if use_responses:
                    stream = await self._await_with_timeout(
                        client.responses.create(**kwargs),
                        phase="stream request",
                    )
                    final = await self._collect_responses_stream(stream, effective_delta)
                    if final is None:
                        raise RuntimeError(
                            "Responses API stream ended without a response.completed event"
                        )
                    response = self._parse_responses(final)
                else:
                    stream = await self._await_with_timeout(
                        client.chat.completions.create(**kwargs),
                        phase="stream request",
                    )
                    chunks = await self._collect_stream_chunks(stream, effective_delta)
                    response = self._parse_chunks(chunks)
                self._record_key_success(key_index)
                if attempt > 0 and self._api_keys:
                    logger.info(
                        "Provider stream recovered on {} after {} attempt(s)",
                        self._key_label(key_index),
                        attempt + 1,
                    )
                return response
            except Exception as e:
                last_error = e
                if delivered_any:
                    # Already streamed content to the user — rotating to another
                    # key would concatenate a second response on top of the first.
                    if self._is_rate_limit_error(e) and self._api_keys:
                        self._record_key_failure(key_index, e)
                    return self._handle_error(e)
                has_remaining_keys = attempt + 1 < max_attempts
                # Timeouts are transient — rotate to next key if available, otherwise
                # let chat_stream_with_retry handle the retry (not a rate-limit failure).
                if isinstance(e, TimeoutError):
                    if has_remaining_keys and self._api_keys:
                        cooldown_s = self._record_key_failure(key_index, e)
                        self._log_rotation(key_index or 0, key_order[attempt + 1], e, cooldown_s)
                        continue
                    return self._handle_error(e)
                if has_remaining_keys and self._is_rate_limit_error(e) and self._api_keys:
                    cooldown_s = self._record_key_failure(key_index, e)
                    self._log_rotation(key_index or 0, key_order[attempt + 1], e, cooldown_s)
                    continue
                if self._is_rate_limit_error(e) and self._api_keys:
                    self._record_key_failure(key_index, e)
                    break
                return self._handle_error(e)
        if last_error and self._is_rate_limit_error(last_error) and self._api_keys:
            logger.error(
                "All configured API keys were rate-limited or out of quota after trying {} keys ({}); last error: {}",
                len(self._api_keys),
                ", ".join(attempted_labels),
                self._error_summary(last_error),
            )
            return LLMResponse(
                content=(
                    "Error: All configured API keys were rate-limited or out of quota "
                    f"after trying {len(self._api_keys)} keys. Last error: {self._error_summary(last_error)}"
                ),
                finish_reason="error",
            )
        return self._handle_error(last_error or RuntimeError("All API keys exhausted"))

    def get_default_model(self) -> str:
        return self.default_model
