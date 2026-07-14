"""Base LLM provider interface."""

import asyncio
import json
import os
import re
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class ToolCallRequest:
    """A tool call request from the LLM."""
    id: str
    name: str
    arguments: dict[str, Any]
    extra_content: dict[str, Any] | None = None
    provider_specific_fields: dict[str, Any] | None = None
    function_provider_specific_fields: dict[str, Any] | None = None

    def to_openai_tool_call(self) -> dict[str, Any]:
        """Serialize to an OpenAI-style tool_call payload."""
        tool_call = {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False),
            },
        }
        if self.extra_content:
            tool_call["extra_content"] = self.extra_content
        if self.provider_specific_fields:
            tool_call["provider_specific_fields"] = self.provider_specific_fields
        if self.function_provider_specific_fields:
            tool_call["function"]["provider_specific_fields"] = self.function_provider_specific_fields
        return tool_call


@dataclass
class LLMResponse:
    """Response from an LLM provider."""
    content: str | None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)
    retry_after: float | None = None  # Provider-supplied retry wait in seconds.
    reasoning_content: str | None = None  # Kimi, DeepSeek-R1 etc.
    thinking_blocks: list[dict] | None = None  # Anthropic extended thinking
    # Codex hosted image_generation tool output items (one per generated image).
    image_calls: list[dict[str, Any]] = field(default_factory=list)
    # Structured error metadata for retry classification.
    error_status_code: int | None = None
    error_type: str | None = None  # e.g. "insufficient_quota"
    error_code: str | None = None  # e.g. "rate_limit_exceeded"

    @property
    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls."""
        return len(self.tool_calls) > 0


@dataclass(frozen=True)
class GenerationSettings:
    """Default generation parameters for LLM calls.

    Stored on the provider so every call site inherits the same defaults
    without having to pass temperature / max_tokens / reasoning_effort
    through every layer.  Individual call sites can still override by
    passing explicit keyword arguments to chat() / chat_with_retry().
    """

    temperature: float = 0.7
    max_tokens: int = 4096
    reasoning_effort: str | None = None


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    Implementations should handle the specifics of each provider's API
    while maintaining a consistent interface.
    """

    _CHAT_RETRY_DELAYS = (1, 2, 4)
    _TRANSIENT_ERROR_MARKERS = (
        "429",
        "rate limit",
        "500",
        "502",
        "503",
        "504",
        "overloaded",
        "timeout",
        "timed out",
        "connection",
        "server error",
        "temporarily unavailable",
    )
    # Semantic tokens from provider error JSON that indicate permanent quota exhaustion.
    _NON_RETRYABLE_429_TOKENS = frozenset({
        "insufficient_quota",
        "quota_exceeded",
        "quota_exhausted",
        "billing_hard_limit_reached",
        "insufficient_balance",
        "credit_balance_too_low",
        "billing_not_active",
        "payment_required",
    })
    # Semantic tokens that indicate a transient rate limit (should retry).
    _RETRYABLE_429_TOKENS = frozenset({
        "rate_limit_exceeded",
        "rate_limit_error",
        "too_many_requests",
        "request_limit_exceeded",
        "overloaded_error",
    })
    # Text markers for fallback classification when no structured tokens available.
    _QUOTA_EXHAUSTION_TEXT_MARKERS = (
        "insufficient_quota",
        "insufficient quota",
        "out of quota",
        "quota exceeded",
        "quota exhausted",
        "billing hard limit",
        "billing_hard_limit_reached",
        "billing limit",
        "billing not active",
        "insufficient balance",
        "credit balance too low",
        "payment required",
        "out of credits",
        "exceeded your current quota",
        "all api keys exhausted",
        "all configured api keys were rate-limited or out of quota",
    )
    _RATE_LIMIT_TEXT_MARKERS = (
        "rate limit",
        "rate_limit",
        "too many requests",
        "retry after",
        "try again in",
        "temporarily unavailable",
        "overloaded",
        "concurrency limit",
    )

    _SENTINEL = object()

    def __init__(self, api_key: str | None = None, api_base: str | None = None):
        self.api_key = api_key
        self.api_base = api_base
        self.generation: GenerationSettings = GenerationSettings()

    @staticmethod
    def _sanitize_empty_content(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Sanitize message content: fix empty blocks, strip internal _meta fields."""
        result: list[dict[str, Any]] = []
        for msg in messages:
            content = msg.get("content")

            if isinstance(content, str) and not content:
                clean = dict(msg)
                clean["content"] = "" if (msg.get("role") == "assistant" and msg.get("tool_calls")) else "(empty)"
                result.append(clean)
                continue

            if isinstance(content, list):
                new_items: list[Any] = []
                changed = False
                for item in content:
                    if (
                        isinstance(item, dict)
                        and item.get("type") in ("text", "input_text", "output_text")
                        and not item.get("text")
                    ):
                        changed = True
                        continue
                    if isinstance(item, dict) and "_meta" in item:
                        new_items.append({k: v for k, v in item.items() if k != "_meta"})
                        changed = True
                    else:
                        new_items.append(item)
                # Ensure content lists with image_url blocks also have a text
                # block.  Some providers (e.g. Xiaomi) reject requests where
                # image_url appears without a companion text block, raising
                # "text is not set".
                has_image = any(
                    isinstance(b, dict) and b.get("type") == "image_url"
                    for b in new_items
                )
                has_text = any(
                    isinstance(b, dict)
                    and b.get("type") in ("text", "input_text", "output_text")
                    for b in new_items
                )
                if has_image and not has_text:
                    new_items.insert(0, {"type": "text", "text": ""})
                    changed = True
                if changed:
                    clean = dict(msg)
                    if new_items:
                        clean["content"] = new_items
                    elif msg.get("role") == "assistant" and msg.get("tool_calls"):
                        clean["content"] = ""
                    else:
                        clean["content"] = "(empty)"
                    result.append(clean)
                    continue

            if isinstance(content, dict):
                clean = dict(msg)
                clean["content"] = [content]
                result.append(clean)
                continue

            result.append(msg)
        return result

    @staticmethod
    def _sanitize_request_messages(
        messages: list[dict[str, Any]],
        allowed_keys: frozenset[str],
    ) -> list[dict[str, Any]]:
        """Keep only provider-safe message keys and normalize assistant content."""
        sanitized = []
        for msg in messages:
            clean = {k: v for k, v in msg.items() if k in allowed_keys}
            if clean.get("role") == "assistant" and "content" not in clean:
                clean["content"] = None
            sanitized.append(clean)
        return sanitized

    @abstractmethod
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
        """
        Send a chat completion request.
        
        Args:
            messages: List of message dicts with 'role' and 'content'.
            tools: Optional list of tool definitions.
            model: Model identifier (provider-specific).
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.
            tool_choice: Tool selection strategy ("auto", "required", or specific tool dict).
        
        Returns:
            LLMResponse with content and/or tool calls.
        """
        pass

    @classmethod
    def _is_transient_error(cls, content: str | None) -> bool:
        err = (content or "").lower()
        # If it looks like a permanent quota exhaustion, don't retry.
        if any(marker in err for marker in cls._QUOTA_EXHAUSTION_TEXT_MARKERS):
            return False
        return any(marker in err for marker in cls._TRANSIENT_ERROR_MARKERS)

    @classmethod
    def _is_transient_response(cls, response: LLMResponse) -> bool:
        """Use structured error metadata when available, fall back to text markers."""
        # Check structured error tokens first (most reliable).
        tokens = {t for t in (response.error_type, response.error_code) if t}
        if tokens:
            if any(t in cls._NON_RETRYABLE_429_TOKENS for t in tokens):
                return False
            if any(t in cls._RETRYABLE_429_TOKENS for t in tokens):
                return True

        # Check status code.
        if response.error_status_code is not None:
            status = response.error_status_code
            if status == 429:
                return cls._is_retryable_429_text(response.content)
            if status >= 500:
                return True

        # Fall back to text-based classification.
        return cls._is_transient_error(response.content)

    @classmethod
    def _is_retryable_429_text(cls, content: str | None) -> bool:
        """Classify a 429 by error message text. Defaults to retryable (rate limit)."""
        err = (content or "").lower()
        if any(marker in err for marker in cls._QUOTA_EXHAUSTION_TEXT_MARKERS):
            return False
        # Unknown 429 defaults to retryable (rate limit).
        return True

    @classmethod
    def _is_quota_exhaustion(cls, content: str | None) -> bool:
        """True when error indicates permanent quota exhaustion (not transient rate limit)."""
        err = (content or "").lower()
        # Check for quota exhaustion markers first.
        has_quota_marker = any(marker in err for marker in cls._QUOTA_EXHAUSTION_TEXT_MARKERS)
        if not has_quota_marker:
            return False
        # "all configured api keys" is a definitive exhaustion signal even
        # though the message text also contains "rate-limited".
        if "all configured api keys" in err or "all api keys exhausted" in err:
            return True
        # For other messages, if rate limit markers are present, it's transient.
        if any(marker in err for marker in cls._RATE_LIMIT_TEXT_MARKERS):
            return False
        return True

    @classmethod
    def _is_quota_exhaustion_response(cls, response: LLMResponse) -> bool:
        """Structured check: is this response a permanent quota exhaustion?"""
        tokens = {t for t in (response.error_type, response.error_code) if t}
        if tokens:
            return any(t in cls._NON_RETRYABLE_429_TOKENS for t in tokens)
        return cls._is_quota_exhaustion(response.content)

    @classmethod
    def _is_rate_limit_response(cls, response: LLMResponse) -> bool:
        """True when the response indicates a transient rate limit (not quota exhaustion)."""
        if response.error_status_code == 429 and not cls._is_quota_exhaustion_response(response):
            return True
        content = (response.content or "").lower()
        has_rate_limit = any(marker in content for marker in cls._RATE_LIMIT_TEXT_MARKERS)
        has_quota = any(marker in content for marker in cls._QUOTA_EXHAUSTION_TEXT_MARKERS)
        return has_rate_limit and not has_quota

    @staticmethod
    def _extract_retry_after(content: str | None) -> float | None:
        """Parse retry-after timing from error message text."""
        text = (content or "").lower()
        patterns = (
            r"retry after\s+(\d+(?:\.\d+)?)\s*(ms|milliseconds|s|sec|secs|seconds|m|min|minutes)?",
            r"try again in\s+(\d+(?:\.\d+)?)\s*(ms|milliseconds|s|sec|secs|seconds|m|min|minutes)",
            r"retry[_-]?after[\"'\s:=]+(\d+(?:\.\d+)?)",
        )
        for idx, pattern in enumerate(patterns):
            match = re.search(pattern, text)
            if not match:
                continue
            value = float(match.group(1))
            unit = match.group(2) if match.lastindex >= 2 else "s"
            if unit and unit in ("ms", "milliseconds"):
                return max(0.1, value / 1000.0)
            if unit and unit in ("m", "min", "minutes"):
                return max(0.1, value * 60.0)
            return max(0.1, value)
        return None

    @staticmethod
    def _extract_retry_after_from_headers(headers: Any) -> float | None:
        """Extract retry wait time from HTTP response headers."""
        if not headers:
            return None
        def _get(name: str) -> Any:
            if hasattr(headers, "get"):
                return headers.get(name) or headers.get(name.title())
            return None
        try:
            retry_ms = _get("retry-after-ms")
            if retry_ms is not None:
                value = float(retry_ms) / 1000.0
                if value > 0:
                    return value
        except (TypeError, ValueError):
            pass
        try:
            retry_s = _get("retry-after")
            if retry_s is not None:
                return max(0.1, float(retry_s))
        except (TypeError, ValueError):
            pass
        return None

    # Markers that indicate the provider/proxy rejected the request because the
    # payload was too large, rather than because the model lacks vision support.
    _PAYLOAD_TOO_LARGE_MARKERS = (
        "request entity too large",
        "payload too large",
        "413",
        "request body too large",
        "content too large",
        "max request size",
    )

    # How many of the most-recent image_url blocks to retain in conversation
    # history before sending to the provider. Older images are replaced with a
    # text placeholder to keep the wire payload under typical proxy limits
    # (e.g. GitHub Copilot's OpenAI-compatible endpoint rejects payloads above
    # a few MB). Set to a higher value if your provider tolerates large
    # multi-image requests, or 0 to keep all images.
    _DEFAULT_IMAGE_HISTORY_KEEP = 1
    try:
        _IMAGE_HISTORY_KEEP = int(
            os.environ.get("NANOBOT_IMAGE_HISTORY_KEEP", _DEFAULT_IMAGE_HISTORY_KEEP)
        )
        if _IMAGE_HISTORY_KEEP < 0:
            _IMAGE_HISTORY_KEEP = _DEFAULT_IMAGE_HISTORY_KEEP
    except (TypeError, ValueError):
        _IMAGE_HISTORY_KEEP = _DEFAULT_IMAGE_HISTORY_KEEP

    @classmethod
    def _is_payload_too_large(cls, content: str | None) -> bool:
        err = (content or "").lower()
        return any(marker in err for marker in cls._PAYLOAD_TOO_LARGE_MARKERS)

    @staticmethod
    def _strip_image_content(
        messages: list[dict[str, Any]],
        *,
        reason: str = "vision",
    ) -> list[dict[str, Any]] | None:
        """Replace image_url blocks with an explicit warning. Returns None if no images found.

        ``reason`` controls the warning text injected so the model knows why it
        cannot see the image and must not fabricate visual verification.
        """
        if reason == "payload_too_large":
            note = (
                "[IMAGE DROPPED — payload exceeded provider/proxy size limit. "
                "You did NOT see this image. Do not claim visual verification; "
                "ask for a smaller capture or use a non-visual confirmation path.]"
            )
        else:
            note = (
                "[IMAGE DROPPED — the active model/provider rejected image content. "
                "You did NOT see this image. Do not claim visual verification; "
                "switch to a vision-capable model or use a non-visual confirmation path.]"
            )

        found = False
        result = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                new_content = []
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "image_url":
                        path = (b.get("_meta") or {}).get("path", "")
                        placeholder = f"{note} (path={path})" if path else note
                        new_content.append({"type": "text", "text": placeholder})
                        found = True
                    else:
                        new_content.append(b)
                result.append({**msg, "content": new_content})
            else:
                result.append(msg)
        return result if found else None

    async def _describe_images_via_fallback(
        self,
        messages: list[dict[str, Any]],
        *,
        reason: str = "vision",
    ) -> list[dict[str, Any]] | None:
        """Describe images via a vision-capable fallback model.

        When the primary model rejects image content, subclasses can override
        this to call a secondary vision model to describe each image, then
        replace the ``image_url`` blocks with text descriptions so the primary
        model can proceed with text-only context.

        Returns modified messages with images replaced by text descriptions,
        or ``None`` if no fallback is available / the fallback call failed
        (caller should then fall back to :meth:`_strip_image_content`).
        """
        return None

    @classmethod
    def _prune_old_images(
        cls,
        messages: list[dict[str, Any]],
        keep: int | None = None,
    ) -> list[dict[str, Any]]:
        """Keep only the most recent ``keep`` image_url blocks; replace older ones with text.

        Walks messages newest-first, counts ``image_url`` blocks, and replaces
        any beyond the keep-budget with a short placeholder so the agent can
        still see *that* an earlier screenshot existed (and find it on disk via
        the original ``_meta.path``). This prevents per-turn payload growth
        when an agent takes many screenshots in a row.

        Returns a new list if anything was pruned, otherwise the original list
        unchanged (caller can safely reuse).
        """
        budget = cls._IMAGE_HISTORY_KEEP if keep is None else keep
        if budget < 0 or not messages:
            return messages

        # First pass: count total image_url blocks.
        total = 0
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "image_url":
                        total += 1
        if total <= budget:
            return messages

        to_drop = total - budget  # number of *oldest* images to replace
        if to_drop <= 0:
            return messages

        dropped = 0
        new_messages: list[dict[str, Any]] = []
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                new_messages.append(msg)
                continue
            changed = False
            new_content: list[Any] = []
            for b in content:
                if (
                    dropped < to_drop
                    and isinstance(b, dict)
                    and b.get("type") == "image_url"
                ):
                    path = (b.get("_meta") or {}).get("path", "")
                    if path:
                        placeholder = (
                            f"[earlier screenshot omitted from history — only the "
                            f"{budget} most recent image(s) are retained. "
                            f"Original path: {path}. Re-capture if you need to see it again.]"
                        )
                    else:
                        placeholder = (
                            f"[earlier screenshot omitted from history — only the "
                            f"{budget} most recent image(s) are retained.]"
                        )
                    new_content.append({"type": "text", "text": placeholder})
                    dropped += 1
                    changed = True
                else:
                    new_content.append(b)
            if changed:
                new_messages.append({**msg, "content": new_content})
            else:
                new_messages.append(msg)
        return new_messages

    async def _safe_chat(self, **kwargs: Any) -> LLMResponse:
        """Call chat() and convert unexpected exceptions to error responses."""
        try:
            return await self.chat(**kwargs)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return LLMResponse(content=f"Error calling LLM: {exc}", finish_reason="error")

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
        """Stream a chat completion, calling *on_content_delta* for each text chunk.

        Returns the same ``LLMResponse`` as :meth:`chat`.  The default
        implementation falls back to a non-streaming call and delivers the
        full content as a single delta.  Providers that support native
        streaming should override this method.
        """
        response = await self.chat(
            messages=messages, tools=tools, model=model,
            max_tokens=max_tokens, temperature=temperature,
            reasoning_effort=reasoning_effort, tool_choice=tool_choice,
        )
        if on_content_delta and response.content:
            await on_content_delta(response.content)
        return response

    async def _safe_chat_stream(self, **kwargs: Any) -> LLMResponse:
        """Call chat_stream() and convert unexpected exceptions to error responses."""
        try:
            return await self.chat_stream(**kwargs)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return LLMResponse(content=f"Error calling LLM: {exc}", finish_reason="error")

    async def chat_stream_with_retry(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: object = _SENTINEL,
        temperature: object = _SENTINEL,
        reasoning_effort: object = _SENTINEL,
        tool_choice: str | dict[str, Any] | None = None,
        on_content_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> LLMResponse:
        """Call chat_stream() with retry on transient provider failures."""
        if max_tokens is self._SENTINEL:
            max_tokens = self.generation.max_tokens
        if temperature is self._SENTINEL:
            temperature = self.generation.temperature
        if reasoning_effort is self._SENTINEL:
            reasoning_effort = self.generation.reasoning_effort

        # Sliding window: drop all but the most recent N images from history
        # before sending. Prevents per-turn payload growth from accumulated
        # screenshots blowing past provider/proxy size limits.
        messages = self._prune_old_images(messages)

        delivered_any = False
        effective_delta = on_content_delta
        if on_content_delta is not None:
            user_delta = on_content_delta

            async def _tracked_delta(text: str) -> None:
                nonlocal delivered_any
                delivered_any = True
                await user_delta(text)

            effective_delta = _tracked_delta

        kw: dict[str, Any] = dict(
            messages=messages, tools=tools, model=model,
            max_tokens=max_tokens, temperature=temperature,
            reasoning_effort=reasoning_effort, tool_choice=tool_choice,
            on_content_delta=effective_delta,
        )

        for attempt, delay in enumerate(self._CHAT_RETRY_DELAYS, start=1):
            response = await self._safe_chat_stream(**kw)

            if response.finish_reason != "error":
                return response

            if delivered_any:
                return response

            if not self._is_transient_response(response):
                too_large = self._is_payload_too_large(response.content)
                strip_reason = "payload_too_large" if too_large else "vision"
                # Try vision fallback first: describe images via a
                # vision-capable model so the primary model can proceed
                # with text descriptions instead of being stripped blind.
                described = await self._describe_images_via_fallback(messages, reason=strip_reason)
                if described is not None:
                    logger.info(
                        "Provider {} (model={}) rejected image content; "
                        "described images via vision fallback model, retrying "
                        "with text descriptions.",
                        type(self).__name__, model or getattr(self, "default_model", None) or "<unknown>",
                    )
                    return await self._safe_chat_stream(**{**kw, "messages": described})
                stripped = self._strip_image_content(messages, reason=strip_reason)
                if stripped is not None:
                    if too_large:
                        logger.error(
                            "Provider {} (model={}) rejected request: payload too large. "
                            "The image is being stripped and the agent will be told it cannot "
                            "see it. Reduce screenshot resolution / inline-byte cap, or send "
                            "fewer prior images. Original error: {}",
                            type(self).__name__, model or getattr(self, "default_model", None) or "<unknown>",
                            (response.content or "")[:300],
                        )
                    else:
                        logger.error(
                            "Provider {} (model={}) rejected request containing image content. "
                            "This usually means the provider/model does not accept vision input, "
                            "or the request shape is wrong. Original error: {}. "
                            "Retrying once with images stripped so the agent can proceed; "
                            "switch to a vision-capable model to enable screenshots.",
                            type(self).__name__, model or getattr(self, "default_model", None) or "<unknown>",
                            (response.content or "")[:300],
                        )
                    return await self._safe_chat_stream(**{**kw, "messages": stripped})
                return response

            # Honor provider-supplied retry delay if available.
            effective_delay = response.retry_after or delay
            logger.warning(
                "LLM transient error (attempt {}/{}), retrying in {:.1f}s: {}",
                attempt, len(self._CHAT_RETRY_DELAYS), effective_delay,
                (response.content or "")[:120].lower(),
            )
            await asyncio.sleep(effective_delay)

        return await self._safe_chat_stream(**kw)

    async def chat_with_retry(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: object = _SENTINEL,
        temperature: object = _SENTINEL,
        reasoning_effort: object = _SENTINEL,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Call chat() with retry on transient provider failures.

        Parameters default to ``self.generation`` when not explicitly passed,
        so callers no longer need to thread temperature / max_tokens /
        reasoning_effort through every layer.
        """
        if max_tokens is self._SENTINEL:
            max_tokens = self.generation.max_tokens
        if temperature is self._SENTINEL:
            temperature = self.generation.temperature
        if reasoning_effort is self._SENTINEL:
            reasoning_effort = self.generation.reasoning_effort

        # Sliding window: drop all but the most recent N images from history.
        messages = self._prune_old_images(messages)

        kw: dict[str, Any] = dict(
            messages=messages, tools=tools, model=model,
            max_tokens=max_tokens, temperature=temperature,
            reasoning_effort=reasoning_effort, tool_choice=tool_choice,
        )

        for attempt, delay in enumerate(self._CHAT_RETRY_DELAYS, start=1):
            response = await self._safe_chat(**kw)

            if response.finish_reason != "error":
                return response

            if not self._is_transient_response(response):
                too_large = self._is_payload_too_large(response.content)
                strip_reason = "payload_too_large" if too_large else "vision"
                # Try vision fallback first: describe images via a
                # vision-capable model so the primary model can proceed
                # with text descriptions instead of being stripped blind.
                described = await self._describe_images_via_fallback(messages, reason=strip_reason)
                if described is not None:
                    logger.info(
                        "Provider {} (model={}) rejected image content; "
                        "described images via vision fallback model, retrying "
                        "with text descriptions.",
                        type(self).__name__, model or getattr(self, "default_model", None) or "<unknown>",
                    )
                    return await self._safe_chat_stream(**{**kw, "messages": described})
                stripped = self._strip_image_content(messages, reason=strip_reason)
                if stripped is not None:
                    if too_large:
                        logger.error(
                            "Provider {} (model={}) rejected request: payload too large. "
                            "The image is being stripped and the agent will be told it cannot "
                            "see it. Reduce screenshot resolution / inline-byte cap, or send "
                            "fewer prior images. Original error: {}",
                            type(self).__name__, model or getattr(self, "default_model", None) or "<unknown>",
                            (response.content or "")[:300],
                        )
                    else:
                        logger.error(
                            "Provider {} (model={}) rejected request containing image content. "
                            "This usually means the provider/model does not accept vision input, "
                            "or the request shape is wrong. Original error: {}. "
                            "Retrying once with images stripped so the agent can proceed; "
                            "switch to a vision-capable model to enable screenshots.",
                            type(self).__name__, model or getattr(self, "default_model", None) or "<unknown>",
                            (response.content or "")[:300],
                        )
                    return await self._safe_chat(**{**kw, "messages": stripped})
                return response

            effective_delay = response.retry_after or delay
            logger.warning(
                "LLM transient error (attempt {}/{}), retrying in {:.1f}s: {}",
                attempt, len(self._CHAT_RETRY_DELAYS), effective_delay,
                (response.content or "")[:120].lower(),
            )
            await asyncio.sleep(effective_delay)

        return await self._safe_chat(**kw)

    @abstractmethod
    def get_default_model(self) -> str:
        """Get the default model for this provider."""
        pass
