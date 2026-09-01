"""Safe text extraction and compaction for structured message content.

Tool results can be either ordinary strings or provider-native content blocks.
The latter may contain large base64 media payloads, so callers must not use
``str(content)`` or string methods on them directly.  This module deliberately
keeps only bounded textual fields and descriptive placeholders for media.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

_DATA_URI_RE = re.compile(
    r"data:[^\s;,]+(?:;[^\s;,]+)*;base64,[A-Za-z0-9+/=_-]+",
    re.IGNORECASE,
)
_LONG_BASE64_RE = re.compile(
    r"(?<![A-Za-z0-9+/=_-])[A-Za-z0-9+/=_-]{48,}(?![A-Za-z0-9+/=_-])"
)

_TEXT_TYPES = frozenset({"text", "input_text", "output_text", "plain_text"})
_IMAGE_TYPES = frozenset({"image", "image_url", "input_image", "output_image"})
_AUDIO_TYPES = frozenset({"audio", "audio_url", "input_audio", "output_audio"})
_VIDEO_TYPES = frozenset({"video", "video_url", "input_video", "output_video"})


def _redact_encoded_text(value: str) -> str:
    """Remove data URIs and long encoded runs from otherwise useful text."""

    value = _DATA_URI_RE.sub("[embedded media omitted]", value)
    return _LONG_BASE64_RE.sub("[encoded content omitted]", value)


def _bounded_text(value: Any, *, max_chars: int = 4000) -> str:
    if not isinstance(value, str):
        return ""
    text = _redact_encoded_text(value).strip()
    if len(text) > max_chars:
        return text[:max_chars].rstrip() + "..."
    return text


def _media_placeholder(kind: str, block: Mapping[str, Any]) -> str:
    """Describe media without exposing its URL or encoded payload."""

    label = ""
    for key in ("alt", "label", "description", "transcript", "title", "name"):
        candidate = _bounded_text(block.get(key), max_chars=240)
        if candidate:
            label = candidate
            break
    return f"[{kind} content omitted]" + (f" {label}" if label else "")


def _block_text(block: Mapping[str, Any]) -> str:
    raw_type = block.get("type")
    block_type = raw_type.strip().lower() if isinstance(raw_type, str) else ""

    if block_type in _TEXT_TYPES:
        text = _bounded_text(block.get("text"))
        if text:
            return text
        return ""
    if block_type in _IMAGE_TYPES:
        return _media_placeholder("image", block)
    if block_type in _AUDIO_TYPES:
        return _media_placeholder("audio", block)
    if block_type in _VIDEO_TYPES:
        return _media_placeholder("video", block)

    # Some adapters omit ``type`` while still using native media fields.
    if any(key in block for key in ("image_url", "image", "image_data")):
        return _media_placeholder("image", block)
    if any(key in block for key in ("input_audio", "audio", "audio_url", "audio_data")):
        return _media_placeholder("audio", block)

    # Preserve an explicitly textual unknown block, but never recursively
    # stringify arbitrary mappings that may contain opaque payloads.
    for key in (
        "text",
        "value",
        "label",
        "description",
        "title",
        "name",
        "message",
        "error",
        "stderr",
        "stdout",
        "output",
    ):
        text = _bounded_text(block.get(key), max_chars=4000)
        if text:
            return text

    if block_type:
        safe_type = re.sub(r"[^a-zA-Z0-9_.-]", "", block_type)[:80] or "unknown"
        return f"[{safe_type} content omitted]"
    return "[structured content omitted]"


def content_to_text(content: Any, *, max_chars: int = 12000) -> str:
    """Extract safe text/placeholders from string, structured, or media content."""

    if content is None:
        return ""
    if isinstance(content, str):
        return _bounded_text(content, max_chars=max_chars)
    if isinstance(content, (bytes, bytearray, memoryview)):
        return "[binary content omitted]"
    if isinstance(content, Mapping):
        text = _block_text(content)
    elif isinstance(content, Sequence) and not isinstance(content, (str, bytes, bytearray)):
        parts: list[str] = []
        for item in content:
            if isinstance(item, Mapping):
                text = _block_text(item)
            elif isinstance(item, str):
                text = _bounded_text(item)
            elif isinstance(item, (bytes, bytearray, memoryview)):
                text = "[binary content omitted]"
            else:
                text = "[structured content omitted]"
            if text:
                parts.append(text)
        text = "\n".join(parts)
    else:
        # Do not call repr/str on arbitrary tool values.  Type information is
        # enough to retain a useful bounded trace without leaking its content.
        text = f"[{type(content).__name__} content omitted]"

    if len(text) > max_chars:
        return text[:max_chars].rstrip() + "..."
    return text


def compact_content(
    content: Any,
    *,
    compacted_placeholder: str = "[compacted to save context]",
    cleared_placeholder: str = "[cleared to save context]",
    head_lines: int = 6,
    tail_lines: int = 4,
    max_chars: int = 700,
) -> str:
    """Return a bounded compacted representation of arbitrary message content."""

    text = content_to_text(content).strip()
    if not text:
        return cleared_placeholder
    if text.startswith(compacted_placeholder) or text == cleared_placeholder:
        return text

    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if not lines:
        lines = [text]
    if len(lines) <= head_lines + tail_lines:
        snippet = "\n".join(lines)
    else:
        snippet = "\n".join([*lines[:head_lines], "...", *lines[-tail_lines:]])

    if len(snippet) > max_chars:
        head = snippet[: int(max_chars * 0.7)].rstrip()
        tail = snippet[-int(max_chars * 0.2):].lstrip()
        snippet = f"{head}\n...\n{tail}"
    return f"{compacted_placeholder}\n{snippet}"


__all__ = ["compact_content", "content_to_text"]
