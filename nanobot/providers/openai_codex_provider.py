"""OpenAI Codex Responses Provider."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable
from typing import Any, AsyncGenerator

import httpx
from loguru import logger

from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from nanobot.providers.codex_auth import get_token as get_codex_token

DEFAULT_CODEX_URL = "https://chatgpt.com/backend-api/codex/responses"
DEFAULT_ORIGINATOR = "codex_cli_rs"
_MAX_CODEX_ID_LEN = 64


class OpenAICodexProvider(LLMProvider):
    """Use Codex OAuth to call the Responses API."""

    def __init__(
        self,
        default_model: str = "openai-codex/gpt-5.1-codex",
        workspace: str | None = None,
    ):
        super().__init__(api_key=None, api_base=None)
        self.default_model = default_model
        self._workspace = workspace

    async def _call_codex(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        model: str | None,
        reasoning_effort: str | None,
        tool_choice: str | dict[str, Any] | None,
        on_content_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> LLMResponse:
        """Shared request logic for both chat() and chat_stream()."""
        model = model or self.default_model
        system_prompt, input_items = _convert_messages(messages)

        token = await asyncio.to_thread(get_codex_token)
        headers = _build_headers(token.account_id, token.access)

        body: dict[str, Any] = {
            "model": _strip_model_prefix(model),
            "store": False,
            "stream": True,
            "instructions": system_prompt,
            "input": input_items,
            "text": {"verbosity": "medium"},
            "include": ["reasoning.encrypted_content"],
            "prompt_cache_key": _prompt_cache_key(messages),
            "tool_choice": tool_choice or "auto",
            "parallel_tool_calls": True,
        }
        if reasoning_effort:
            body["reasoning"] = {"effort": reasoning_effort}
        body["tools"] = _convert_tools(tools) if tools else []

        try:
            try:
                content, tool_calls, finish_reason, image_calls = await _request_codex(
                    DEFAULT_CODEX_URL, headers, body, verify=True,
                    on_content_delta=on_content_delta,
                )
            except Exception as e:
                if "CERTIFICATE_VERIFY_FAILED" not in str(e):
                    raise
                logger.warning("SSL verification failed for Codex API; retrying with verify=False")
                content, tool_calls, finish_reason, image_calls = await _request_codex(
                    DEFAULT_CODEX_URL, headers, body, verify=False,
                    on_content_delta=on_content_delta,
                )
            return LLMResponse(
                content=content,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
                image_calls=image_calls,
            )
        except _CodexHTTPError as e:
            msg = f"Error calling Codex: {e}"
            retry_after = e.retry_after or self._extract_retry_after(msg)
            return LLMResponse(
                content=msg, finish_reason="error",
                retry_after=retry_after,
                error_status_code=e.status_code,
                error_type=e.error_type,
                error_code=e.error_code,
            )
        except Exception as e:
            msg = f"Error calling Codex: {e}"
            retry_after = self._extract_retry_after(msg)
            return LLMResponse(content=msg, finish_reason="error", retry_after=retry_after)

    async def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
        model: str | None = None, max_tokens: int = 4096, temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        return await self._call_codex(messages, tools, model, reasoning_effort, tool_choice)

    async def chat_stream(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
        model: str | None = None, max_tokens: int = 4096, temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        on_content_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> LLMResponse:
        return await self._call_codex(messages, tools, model, reasoning_effort, tool_choice, on_content_delta)

    def get_default_model(self) -> str:
        return self.default_model


def _strip_model_prefix(model: str) -> str:
    if model.startswith("openai-codex/") or model.startswith("openai_codex/"):
        return model.split("/", 1)[1]
    return model


def _normalize_codex_id(value: Any, prefix: str) -> str:
    """Return a Codex-safe opaque identifier."""
    if isinstance(value, str):
        text = value.strip()
        if text and len(text) <= _MAX_CODEX_ID_LEN and all(ch.isalnum() or ch in "_-" for ch in text):
            return text
        if text:
            digest_len = max(8, _MAX_CODEX_ID_LEN - len(prefix))
            return f"{prefix}{hashlib.sha1(text.encode('utf-8')).hexdigest()[:digest_len]}"
    return f"{prefix}0"


def _normalize_tool_call_ref(
    tool_call_id: Any,
    *,
    fallback_item_id: str | None = None,
) -> tuple[str, str | None]:
    """Normalize a stored tool-call reference into Codex-safe call/item IDs."""
    call_id, item_id = _split_tool_call_id(tool_call_id)
    safe_call_id = _normalize_codex_id(call_id, "call_")
    if item_id:
        return safe_call_id, _normalize_codex_id(item_id, "fc_")
    if fallback_item_id:
        return safe_call_id, _normalize_codex_id(fallback_item_id, "fc_")
    return safe_call_id, None


def _build_headers(account_id: str, token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "ChatGPT-Account-ID": account_id,
        "OpenAI-Beta": "responses=experimental",
        "originator": DEFAULT_ORIGINATOR,
        "User-Agent": f"{DEFAULT_ORIGINATOR}/0.0.1 (python)",
        "accept": "text/event-stream",
        "content-type": "application/json",
    }


class _CodexHTTPError(RuntimeError):
    """Carries structured error metadata from a Codex HTTP error response."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        retry_after: float | None = None,
        error_type: str | None = None,
        error_code: str | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after
        self.error_type = error_type
        self.error_code = error_code


async def _request_codex(
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    verify: bool,
    on_content_delta: Callable[[str], Awaitable[None]] | None = None,
) -> tuple[str, list[ToolCallRequest], str, list[dict[str, Any]]]:
    async with httpx.AsyncClient(timeout=60.0, verify=verify) as client:
        async with client.stream("POST", url, headers=headers, json=body) as response:
            if response.status_code != 200:
                text = await response.aread()
                raw = text.decode("utf-8", "ignore")
                retry_after = LLMProvider._extract_retry_after_from_headers(response.headers)
                error_type, error_code = _extract_error_tokens(raw)
                raise _CodexHTTPError(
                    _friendly_error(response.status_code, raw, error_type),
                    status_code=response.status_code,
                    retry_after=retry_after,
                    error_type=error_type,
                    error_code=error_code,
                )
            return await _consume_sse(response, on_content_delta)


def _convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI function-calling schema to Codex flat format."""
    converted: list[dict[str, Any]] = []
    for tool in tools:
        fn = (tool.get("function") or {}) if tool.get("type") == "function" else tool
        name = fn.get("name")
        if not name:
            continue
        params = fn.get("parameters") or {}
        converted.append({
            "type": "function",
            "name": name,
            "description": fn.get("description") or "",
            "parameters": params if isinstance(params, dict) else {},
        })
    return converted


def _convert_messages(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    system_parts: list[str] = []
    input_items: list[dict[str, Any]] = []

    for idx, msg in enumerate(messages):
        role = msg.get("role")
        content = msg.get("content")

        if role == "system":
            # Collect and merge every system message so later scoped guidance
            # does not replace the agent's identity and tool instructions.
            part = content if isinstance(content, str) else ""
            if part:
                system_parts.append(part)
            continue

        if role == "user":
            input_items.append(_convert_user_message(content))
            continue

        if role == "assistant":
            if isinstance(content, str) and content:
                input_items.append({
                    "type": "message", "role": "assistant",
                    "content": [{"type": "output_text", "text": content}],
                    "status": "completed", "id": f"msg_{idx}",
                })
            for tool_call in msg.get("tool_calls", []) or []:
                fn = tool_call.get("function") or {}
                call_id, item_id = _normalize_tool_call_ref(
                    tool_call.get("id"),
                    fallback_item_id=f"fc_{idx}",
                )
                input_items.append({
                    "type": "function_call",
                    "id": item_id or _normalize_codex_id(f"fc_{idx}", "fc_"),
                    "call_id": call_id or _normalize_codex_id(f"call_{idx}", "call_"),
                    "name": fn.get("name"),
                    "arguments": fn.get("arguments") or "{}",
                })
            continue

        if role == "tool":
            call_id, _ = _normalize_tool_call_ref(msg.get("tool_call_id"))
            text_parts, image_urls = _split_tool_content(content)
            output_text = text_parts if text_parts else ""
            input_items.append({"type": "function_call_output", "call_id": call_id, "output": output_text})
            # The Codex Responses API has no field for image output on
            # function_call_output, so re-attach any tool-returned image(s)
            # as a synthetic user message with input_image parts so the
            # model can actually see them (e.g. desktop_use screenshots).
            if image_urls:
                tool_name = msg.get("name") or ""
                parts: list[dict[str, Any]] = [{
                    "type": "input_text",
                    "text": f"[tool {tool_name} returned the following image(s)]".strip(),
                }]
                for url in image_urls:
                    parts.append({"type": "input_image", "image_url": url, "detail": "auto"})
                input_items.append({"role": "user", "content": parts})

    system_prompt = "\n\n".join(system_parts)
    return system_prompt, input_items


def _split_tool_content(content: Any) -> tuple[str, list[str]]:
    """Split a tool message's content into (text_output, image_urls).

    Mirrors the OpenAI-compat provider's behavior: image_url blocks are
    extracted and returned separately so they can be re-attached as an
    `input_image` synthetic user message, instead of being JSON-dumped
    into the function_call_output text (which is invisible to vision).
    """
    if content is None:
        return "", []
    if isinstance(content, str):
        return content, []
    if isinstance(content, list):
        text_parts: list[str] = []
        image_urls: list[str] = []
        for item in content:
            if isinstance(item, dict):
                t = item.get("type")
                if t == "image_url":
                    iu = item.get("image_url")
                    url = iu if isinstance(iu, str) else (iu or {}).get("url") if isinstance(iu, dict) else None
                    if isinstance(url, str) and url:
                        image_urls.append(url)
                    continue
                if t == "input_image":
                    iu = item.get("image_url")
                    if isinstance(iu, str) and iu:
                        image_urls.append(iu)
                    continue
                text = item.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
                    continue
                text_parts.append(json.dumps(item, ensure_ascii=False))
            elif isinstance(item, str):
                text_parts.append(item)
            else:
                text_parts.append(str(item))
        return "".join(text_parts), image_urls
    return json.dumps(content, ensure_ascii=False), []


def _convert_user_message(content: Any) -> dict[str, Any]:
    if isinstance(content, str):
        return {"role": "user", "content": [{"type": "input_text", "text": content}]}
    if isinstance(content, list):
        converted: list[dict[str, Any]] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                converted.append({"type": "input_text", "text": item.get("text", "")})
            elif item.get("type") == "image_url":
                url = (item.get("image_url") or {}).get("url")
                if url:
                    converted.append({"type": "input_image", "image_url": url, "detail": "auto"})
        if converted:
            return {"role": "user", "content": converted}
    return {"role": "user", "content": [{"type": "input_text", "text": ""}]}


def _split_tool_call_id(tool_call_id: Any) -> tuple[str, str | None]:
    if isinstance(tool_call_id, str) and tool_call_id:
        if "|" in tool_call_id:
            call_id, item_id = tool_call_id.split("|", 1)
            return call_id, item_id or None
        return tool_call_id, None
    return "call_0", None


def _prompt_cache_key(messages: list[dict[str, Any]]) -> str:
    raw = json.dumps(messages, ensure_ascii=True, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


async def _iter_sse(response: httpx.Response) -> AsyncGenerator[dict[str, Any], None]:
    buffer: list[str] = []
    async for line in response.aiter_lines():
        if line == "":
            if buffer:
                data_lines = [l[5:].strip() for l in buffer if l.startswith("data:")]
                buffer = []
                if not data_lines:
                    continue
                data = "\n".join(data_lines).strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    yield json.loads(data)
                except Exception:
                    continue
            continue
        buffer.append(line)


async def _consume_sse(
    response: httpx.Response,
    on_content_delta: Callable[[str], Awaitable[None]] | None = None,
) -> tuple[str, list[ToolCallRequest], str, list[dict[str, Any]]]:
    content = ""
    tool_calls: list[ToolCallRequest] = []
    tool_call_buffers: dict[str, dict[str, Any]] = {}
    finish_reason = "stop"

    async for event in _iter_sse(response):
        event_type = event.get("type")
        if event_type == "response.output_item.added":
            item = event.get("item") or {}
            if item.get("type") == "function_call":
                call_id = item.get("call_id")
                if not call_id:
                    continue
                tool_call_buffers[call_id] = {
                    "id": item.get("id") or "fc_0",
                    "name": item.get("name"),
                    "arguments": item.get("arguments") or "",
                }
        elif event_type == "response.output_text.delta":
            delta_text = event.get("delta") or ""
            content += delta_text
            if on_content_delta and delta_text:
                await on_content_delta(delta_text)
        elif event_type == "response.function_call_arguments.delta":
            call_id = event.get("call_id")
            if call_id and call_id in tool_call_buffers:
                tool_call_buffers[call_id]["arguments"] += event.get("delta") or ""
        elif event_type == "response.function_call_arguments.done":
            call_id = event.get("call_id")
            if call_id and call_id in tool_call_buffers:
                tool_call_buffers[call_id]["arguments"] = event.get("arguments") or ""
        elif event_type == "response.output_item.done":
            item = event.get("item") or {}
            item_type = item.get("type")
            if item_type == "function_call":
                call_id = item.get("call_id")
                if not call_id:
                    continue
                buf = tool_call_buffers.get(call_id) or {}
                args_raw = buf.get("arguments") or item.get("arguments") or "{}"
                try:
                    args = json.loads(args_raw)
                except Exception:
                    args = {"raw": args_raw}
                safe_call_id = _normalize_codex_id(call_id, "call_")
                safe_item_id = _normalize_codex_id(buf.get("id") or item.get("id") or "fc_0", "fc_")
                tool_calls.append(
                    ToolCallRequest(
                        id=f"{safe_call_id}|{safe_item_id}",
                        name=buf.get("name") or item.get("name"),
                        arguments=args,
                    )
                )
        elif event_type == "response.completed":
            status = (event.get("response") or {}).get("status")
            finish_reason = _map_finish_reason(status)
        elif event_type in {"error", "response.failed"}:
            raise RuntimeError("Codex response failed")

    return content, tool_calls, finish_reason, []


_FINISH_REASON_MAP = {"completed": "stop", "incomplete": "length", "failed": "error", "cancelled": "error"}


def _map_finish_reason(status: str | None) -> str:
    return _FINISH_REASON_MAP.get(status or "completed", "stop")


def _extract_error_tokens(raw: str) -> tuple[str | None, str | None]:
    """Parse error type and code from a JSON error response body."""
    try:
        data = json.loads(raw)
    except Exception:
        return None, None
    if not isinstance(data, dict):
        return None, None
    error_obj = data.get("error")
    error_type = data.get("type")
    error_code = data.get("code")
    if isinstance(error_obj, dict):
        error_type = error_obj.get("type") or error_type
        error_code = error_obj.get("code") or error_code
    t = str(error_type).strip().lower() if error_type else None
    c = str(error_code).strip().lower() if error_code else None
    return t or None, c or None


def _friendly_error(status_code: int, raw: str, error_type: str | None = None) -> str:
    if status_code == 429:
        # Distinguish rate limit from quota exhaustion.
        quota_tokens = {"insufficient_quota", "quota_exceeded", "quota_exhausted",
                        "billing_hard_limit_reached", "insufficient_balance"}
        if error_type and error_type in quota_tokens:
            return "ChatGPT quota exhausted. Please check your plan and billing."
        return "ChatGPT rate limit triggered. Please try again shortly."
    return f"HTTP {status_code}: {raw}"
