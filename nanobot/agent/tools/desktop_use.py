"""Stateless pure-Python desktop-control tool.

``desktop_use`` is nanobot's desktop-control tool, modeled after Anthropic's
``computer-use-demo`` and OpenAI's Operator agent:

* Stateless — every call is a one-shot OS-level operation. No session, no
  daemon, no socket, no AX tree.
* Pixel-coordinate first — the model sees a screenshot and clicks at (x, y).
* Tiny action set (``screenshot``, ``mouse_move``, ``click`` family, ``scroll``,
  ``type``, ``key``, ``wait``, ``cursor_position``) modelled on the published
  Operator / Anthropic action vocabulary.
* Optional coordinate downscaling to one of the resolutions that vision models
  are trained on (XGA / WXGA / FWXGA), to save tokens and improve targeting.

This trades the AX tree's structural precision for an architecture with
virtually no failure modes: the only thing that can go wrong is a missing OS
permission, and that surfaces as a clean error from the underlying syscall.

Disabled by default. On macOS the user must install nanobot with the optional
``desktop`` extra (``pip install 'nanobot-ai[desktop]'``) which pulls in the
PyObjC frameworks needed to call CoreGraphics. Accessibility (clicks/typing)
and Screen Recording (screenshots) permissions must also be granted in System
Settings -> Privacy & Security.
"""

from __future__ import annotations

import asyncio
import base64
import json
import math
import os
import platform
import random
import tempfile
import time
from typing import Any

from nanobot.agent.tools.base import Tool


# Cap on the inline screenshot payload we'll attach to a tool result. Larger
# files are skipped (path-only) to stay within provider request limits.
# Default is conservative because many provider proxies (e.g. GitHub Copilot's
# OpenAI-compatible endpoint) cap request bodies well below the model's
# nominal limit and reject larger payloads with HTTP 413. The base64 encoding
# also inflates the wire size by ~33%. Override via env NANOBOT_DESKTOP_USE_MAX_INLINE_BYTES.
_DEFAULT_MAX_INLINE_SCREENSHOT_BYTES = 900 * 1024
try:
    _MAX_INLINE_SCREENSHOT_BYTES = int(
        os.environ.get(
            "NANOBOT_DESKTOP_USE_MAX_INLINE_BYTES",
            _DEFAULT_MAX_INLINE_SCREENSHOT_BYTES,
        )
    )
    if _MAX_INLINE_SCREENSHOT_BYTES <= 0:
        _MAX_INLINE_SCREENSHOT_BYTES = _DEFAULT_MAX_INLINE_SCREENSHOT_BYTES
except (TypeError, ValueError):
    _MAX_INLINE_SCREENSHOT_BYTES = _DEFAULT_MAX_INLINE_SCREENSHOT_BYTES

# Resolutions that vision-capable models tend to be trained on. We downscale
# coordinates to whichever one matches the screen's aspect ratio so the model
# sees screenshots at a familiar size and pixel coordinates round-trip cleanly.
# Source: Anthropic computer-use-demo's MAX_SCALING_TARGETS table.
_SCALING_TARGETS: tuple[tuple[int, int], ...] = (
    (1024, 768),   # XGA   4:3
    (1280, 800),   # WXGA  16:10
    (1366, 768),   # FWXGA ~16:9
)

# Allowed key names for ``key`` / ``hold_key``. Kept loose because the macOS
# backend resolves names via NSEvent's character map; this set is just for
# error messages so we fail fast on obvious typos.
_KEY_ALIASES = {
    "return": "Return", "enter": "Return",
    "tab": "Tab", "escape": "Escape", "esc": "Escape",
    "space": "Space", "backspace": "Backspace", "delete": "Delete",
    "up": "Up", "down": "Down", "left": "Left", "right": "Right",
    "home": "Home", "end": "End", "pageup": "PageUp", "pagedown": "PageDown",
}

# ── Human-like typing model ─────────────────────────────────────────────
# When ``humanize_typing`` is on, inter-key delays are drawn from a log-normal
# distribution around a mean derived from ``typing_delay_ms`` (clamped to a
# realistic floor so even a 12ms config still reads as human). Real typists
# average ~100ms between keys with a long right tail; we shape the jitter
# accordingly and apply small multipliers for common bigrams (muscle memory)
# and a penalty for shifted/punctuation chars (slower to reach).
_HUMAN_MIN_MEAN_MS = 70          # never faster than this even if config asks
_HUMAN_MAX_MEAN_MS = 220         # never slower than this baseline
_HUMAN_JITTER_SIGMA = 0.35       # log-normal sigma — moderate spread
_HUMAN_PAUSE_PROB = 0.04         # 4% chance of a brief "thinking" pause
_HUMAN_PAUSE_MS_LO = 180
_HUMAN_PAUSE_MS_HI = 520
_HUMAN_FIRST_KEY_BONUS_LO = 90   # extra delay before the very first key
_HUMAN_FIRST_KEY_BONUS_HI = 260
_HUMAN_SPACE_MS_BONUS = 30       # spaces are slightly slower than letters

# Bigrams that experienced typists chunk as one motion -> shorter delay.
_FAST_BIGRAMS = frozenset({
    "th", "he", "in", "er", "an", "re", "on", "at", "en", "nd",
    "ti", "es", "or", "te", "of", "ed", "is", "it", "al", "ar",
    "st", "to", "nt", "ng", "se", "ha", "as", "ou", "io", "le",
    "ve", "co", "me", "de", "hi", "ri", "ro", "ic", "ne", "ea",
})


class DesktopUseTool(Tool):
    """Stateless pixel-coordinate desktop driver (Operator / Anthropic style)."""

    def __init__(
        self,
        screenshot_delay: float = 0.5,
        typing_delay_ms: int = 12,
        scaling_enabled: bool = True,
        max_output_chars: int = 12000,
        working_dir: str | None = None,
        humanize_typing: bool = True,
    ):
        self.screenshot_delay = max(0.0, float(screenshot_delay))
        self.typing_delay_ms = max(0, int(typing_delay_ms))
        self.scaling_enabled = bool(scaling_enabled)
        self.max_output_chars = max_output_chars
        self.working_dir = working_dir
        self.humanize_typing = bool(humanize_typing)
        self._backend: _Backend | None = None
        self._backend_error: str | None = None

    # ------------------------------------------------------------------
    # Tool interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "desktop_use"

    @property
    def description(self) -> str:
        return (
            "Drive the local desktop with raw pixel coordinates. Stateless and "
            "daemonless: every call is a one-shot OS event (no session, no AX "
            "tree, no `@eN` refs). Loop: `screenshot` -> reason on the image "
            "-> `mouse_move` -> `left_click` -> `type` / `key` -> `screenshot`. "
            "Coordinates are in the SCALED screenshot space the tool returns; "
            "don't try to read native screen pixels. Always screenshot before "
            "and after a click so each step has a visual trace. To change "
            "apps, screenshot the Dock/taskbar and click the icon, or use "
            "`key` with 'Cmd+Tab' / 'Alt+Tab'. Run `action='info'` to see the "
            "logical screen size."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "info",
                        "screenshot",
                        "cursor_position",
                        "mouse_move",
                        "left_click",
                        "right_click",
                        "middle_click",
                        "double_click",
                        "left_click_drag",
                        "scroll",
                        "type",
                        "key",
                        "hold_key",
                        "wait",
                    ],
                    "description": (
                        "Operation to perform. `info` returns screen size + "
                        "scale. `screenshot` captures the desktop. Coordinate "
                        "actions take `coordinate=[x, y]`."
                    ),
                },
                "coordinate": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                    "description": "[x, y] in scaled screenshot space.",
                },
                "start_coordinate": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                    "description": "Drag origin; required for `left_click_drag`.",
                },
                "text": {
                    "type": "string",
                    "description": (
                        "Text to type (`type`) or key chord to press "
                        "(`key` / `hold_key`, e.g. 'Cmd+S', 'Return')."
                    ),
                },
                "scroll_direction": {
                    "type": "string",
                    "enum": ["up", "down", "left", "right"],
                },
                "scroll_amount": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "description": "Wheel notches (1 ~= 100px on macOS).",
                },
                "duration": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 30,
                    "description": "Seconds for `wait` or `hold_key`.",
                },
            },
            "required": ["action"],
        }

    @property
    def supports_parallel_calls(self) -> bool:
        # Desktop input is global state; serialize calls.
        return False

    async def execute(self, action: str, **kwargs: Any) -> Any:
        backend = self._get_backend()
        if backend is None:
            return self._error(self._backend_error or "desktop backend unavailable")

        try:
            if action == "info":
                return self._ok(backend.info())

            if action == "screenshot":
                payload, image_block = await asyncio.to_thread(self._screenshot, backend)
                if image_block is None and payload.get("inline_skipped_reason"):
                    payload["warning"] = (
                        "IMAGE NOT SENT TO MODEL — "
                        + payload["inline_skipped_reason"]
                        + ". You did NOT see this screenshot. Do not claim visual "
                        "verification; use osascript/exec or a smaller-region capture."
                    )
                text = json.dumps(payload, ensure_ascii=False)
                if image_block is not None:
                    return [{"type": "text", "text": text}, image_block]
                return text

            if action == "cursor_position":
                x, y = backend.cursor_position()
                sx, sy = self._scale_to_model(backend, x, y)
                return self._ok({"x": sx, "y": sy, "raw": {"x": x, "y": y}})

            if action == "wait":
                duration = float(kwargs.get("duration") or 0.0)
                if duration <= 0 or duration > 30:
                    return self._error("duration must be > 0 and <= 30 seconds")
                await asyncio.sleep(duration)
                return self._ok({"waited": duration})

            if action == "mouse_move":
                x, y = self._require_coord(kwargs.get("coordinate"))
                rx, ry = self._scale_to_screen(backend, x, y)
                await asyncio.to_thread(backend.mouse_move, rx, ry)
                return self._ok({"moved_to": [rx, ry]})

            if action in ("left_click", "right_click", "middle_click", "double_click"):
                coord = kwargs.get("coordinate")
                if coord is not None:
                    x, y = self._require_coord(coord)
                    rx, ry = self._scale_to_screen(backend, x, y)
                    await asyncio.to_thread(backend.mouse_move, rx, ry)
                else:
                    rx, ry = backend.cursor_position()
                button = {
                    "left_click": "left",
                    "right_click": "right",
                    "middle_click": "middle",
                    "double_click": "left",
                }[action]
                clicks = 2 if action == "double_click" else 1
                await asyncio.to_thread(backend.mouse_click, rx, ry, button, clicks)
                return self._ok({"clicked": [rx, ry], "button": button, "count": clicks})

            if action == "left_click_drag":
                start = self._require_coord(kwargs.get("start_coordinate"), "start_coordinate")
                end = self._require_coord(kwargs.get("coordinate"))
                sx, sy = self._scale_to_screen(backend, *start)
                ex, ey = self._scale_to_screen(backend, *end)
                await asyncio.to_thread(backend.mouse_drag, sx, sy, ex, ey)
                return self._ok({"dragged": {"from": [sx, sy], "to": [ex, ey]}})

            if action == "scroll":
                direction = kwargs.get("scroll_direction")
                if direction not in ("up", "down", "left", "right"):
                    return self._error("scroll_direction must be up/down/left/right")
                amount = int(kwargs.get("scroll_amount") or 3)
                amount = max(1, min(amount, 100))
                coord = kwargs.get("coordinate")
                if coord is not None:
                    x, y = self._require_coord(coord)
                    rx, ry = self._scale_to_screen(backend, x, y)
                    await asyncio.to_thread(backend.mouse_move, rx, ry)
                await asyncio.to_thread(backend.scroll, direction, amount)
                return self._ok({"scrolled": direction, "amount": amount})

            if action == "type":
                text = kwargs.get("text")
                if not isinstance(text, str) or not text:
                    return self._error("text is required for `type`")
                if len(text) > 4000:
                    return self._error("text too long (max 4000 chars per call)")
                delays = self._human_delays(text) if self.humanize_typing else None
                await asyncio.to_thread(
                    backend.type_text, text, self.typing_delay_ms, delays,
                )
                return self._ok({"typed_chars": len(text)})

            if action == "key":
                chord = kwargs.get("text")
                if not isinstance(chord, str) or not chord:
                    return self._error("text is required for `key` (e.g. 'Cmd+S', 'Return')")
                await asyncio.to_thread(backend.key_chord, chord)
                return self._ok({"pressed": chord})

            if action == "hold_key":
                chord = kwargs.get("text")
                duration = float(kwargs.get("duration") or 0.0)
                if not isinstance(chord, str) or not chord:
                    return self._error("text is required for `hold_key`")
                if duration <= 0 or duration > 10:
                    return self._error("duration must be > 0 and <= 10 seconds")
                await asyncio.to_thread(backend.hold_key, chord, duration)
                return self._ok({"held": chord, "duration": duration})

            return self._error(f"unknown action: {action}")

        except _DesktopError as exc:
            return self._error(str(exc))
        except Exception as exc:  # pragma: no cover - defensive
            return self._error(f"{type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _human_delays(self, text: str) -> list[int]:
        """Per-character inter-key delays (ms) that look like a real typist.

        The first key gets an extra startup latency (hands moving to the
        keyboard). Subsequent delays are drawn from a log-normal around the
        configured mean (clamped to a realistic human band), with bigram
        muscle-memory boosts and a small chance of a brief "thinking" pause.
        Spaces and shifted/punctuation chars get a small extra cost.
        """
        mean = max(
            _HUMAN_MIN_MEAN_MS,
            min(_HUMAN_MAX_MEAN_MS, self.typing_delay_ms if self.typing_delay_ms > 0 else 90),
        )
        n = len(text)
        delays: list[int] = []
        for i in range(n):
            ch = text[i]
            mu = float(mean)
            # Common bigram muscle memory -> faster.
            if i > 0:
                pair = (text[i - 1] + ch).lower()
                if pair in _FAST_BIGRAMS:
                    mu *= 0.55
            # Spaces are slightly slower than letters.
            if ch == " ":
                mu += _HUMAN_SPACE_MS_BONUS
            # Punctuation / digits / uppercase (= shifted) cost more.
            if not ch.isalpha() or ch.isupper():
                mu *= 1.18
            # Log-normal jitter (always positive, long right tail).
            d = self._lognormal_ms(mu)
            # Occasional brief "thinking" pause folded into the same delay
            # so per-char count stays exactly n.
            if i < n - 1 and random.random() < _HUMAN_PAUSE_PROB:
                d = min(2000, d + random.randint(_HUMAN_PAUSE_MS_LO, _HUMAN_PAUSE_MS_HI))
            delays.append(d)
        # Extra startup latency after the very first keystroke (orienting
        # before continuing the rest of the word).
        if delays:
            delays[0] = delays[0] + random.randint(_HUMAN_FIRST_KEY_BONUS_LO, _HUMAN_FIRST_KEY_BONUS_HI)
        return delays

    @staticmethod
    def _lognormal_ms(mean: float) -> int:
        mu_ln = math.log(max(1.0, mean))
        sample = random.lognormvariate(mu_ln, _HUMAN_JITTER_SIGMA)
        return max(12, min(900, int(round(sample))))

    def _get_backend(self) -> "_Backend | None":
        if self._backend is not None:
            return self._backend
        if self._backend_error is not None:
            return None
        system = platform.system()
        try:
            if system == "Darwin":
                self._backend = _MacBackend()
            elif system == "Windows":
                raise _DesktopError(
                    "Windows backend not implemented yet; contribute a backend."
                )
            elif system == "Linux":
                raise _DesktopError(
                    "Linux backend not implemented yet; contribute a backend."
                )
            else:
                raise _DesktopError(f"unsupported platform: {system}")
            return self._backend
        except _DesktopError as exc:
            self._backend_error = str(exc)
            return None

    def _screenshot(self, backend: "_Backend") -> tuple[dict[str, Any], dict[str, Any] | None]:
        if self.screenshot_delay > 0:
            time.sleep(self.screenshot_delay)
        out_dir = self.working_dir or tempfile.gettempdir()
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"desktop_use_{int(time.time() * 1000)}.png")
        native_w, native_h = backend.screen_size()
        backend.screenshot(path)
        # Downscale on disk to the model-friendly resolution.
        scaled_w, scaled_h = self._target_size(native_w, native_h)
        if (scaled_w, scaled_h) != (native_w, native_h):
            backend.resize_png(path, scaled_w, scaled_h)
        payload: dict[str, Any] = {
            "ok": True,
            "path": path,
            "native": {"width": native_w, "height": native_h},
            "scaled": {"width": scaled_w, "height": scaled_h},
        }
        image_block, skip_reason = self._inline_image(path)
        if skip_reason:
            payload["inline_skipped_reason"] = skip_reason
        return payload, image_block

    def _inline_image(self, path: str) -> tuple[dict[str, Any] | None, str | None]:
        try:
            size = os.path.getsize(path)
        except OSError as exc:
            return None, f"could not stat image: {exc}"
        if size <= 0:
            return None, "image file empty"
        if size > _MAX_INLINE_SCREENSHOT_BYTES:
            return None, (
                f"image {size} bytes exceeds inline cap "
                f"{_MAX_INLINE_SCREENSHOT_BYTES} bytes (override with "
                f"NANOBOT_DESKTOP_USE_MAX_INLINE_BYTES env)"
            )
        try:
            with open(path, "rb") as f:
                data = f.read()
        except OSError as exc:
            return None, f"could not read image: {exc}"
        encoded = base64.b64encode(data).decode("ascii")
        return (
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encoded}"},
            },
            None,
        )

    def _target_size(self, w: int, h: int) -> tuple[int, int]:
        if not self.scaling_enabled or w <= 0 or h <= 0:
            return w, h
        ratio = w / h
        for tw, th in _SCALING_TARGETS:
            if abs((tw / th) - ratio) < 0.03 and tw < w:
                return tw, th
        return w, h

    def _scale_to_screen(self, backend: "_Backend", x: int, y: int) -> tuple[int, int]:
        """Map model-space coords -> native screen coords."""
        nw, nh = backend.screen_size()
        sw, sh = self._target_size(nw, nh)
        if (sw, sh) == (nw, nh) or sw <= 0 or sh <= 0:
            return int(x), int(y)
        if x < 0 or y < 0 or x > sw or y > sh:
            raise _DesktopError(
                f"coordinate ({x}, {y}) outside scaled screen {sw}x{sh}"
            )
        return round(x * nw / sw), round(y * nh / sh)

    def _scale_to_model(self, backend: "_Backend", x: int, y: int) -> tuple[int, int]:
        nw, nh = backend.screen_size()
        sw, sh = self._target_size(nw, nh)
        if (sw, sh) == (nw, nh):
            return int(x), int(y)
        return round(x * sw / nw), round(y * sh / nh)

    @staticmethod
    def _require_coord(value: Any, name: str = "coordinate") -> tuple[int, int]:
        if (
            not isinstance(value, (list, tuple))
            or len(value) != 2
            or not all(isinstance(v, (int, float)) for v in value)
        ):
            raise _DesktopError(f"{name} must be [x, y] with two numbers")
        return int(value[0]), int(value[1])

    @staticmethod
    def _ok(payload: dict[str, Any]) -> str:
        return json.dumps({"ok": True, **payload}, ensure_ascii=False)

    @staticmethod
    def _error(message: str) -> str:
        return json.dumps({"ok": False, "error": message}, ensure_ascii=False)


# ----------------------------------------------------------------------
# Backend abstraction
# ----------------------------------------------------------------------


class _DesktopError(Exception):
    """User-visible desktop-control failure."""


class _Backend:
    def info(self) -> dict[str, Any]: ...
    def screen_size(self) -> tuple[int, int]: ...
    def cursor_position(self) -> tuple[int, int]: ...
    def screenshot(self, path: str) -> None: ...
    def resize_png(self, path: str, width: int, height: int) -> None: ...
    def mouse_move(self, x: int, y: int) -> None: ...
    def mouse_click(self, x: int, y: int, button: str, clicks: int) -> None: ...
    def mouse_drag(self, x1: int, y1: int, x2: int, y2: int) -> None: ...
    def scroll(self, direction: str, amount: int) -> None: ...
    def type_text(self, text: str, delay_ms: int, delays_ms: list[int] | None = None) -> None: ...
    def key_chord(self, chord: str) -> None: ...
    def hold_key(self, chord: str, duration: float) -> None: ...


# ----------------------------------------------------------------------
# macOS backend (PyObjC / Quartz / AppKit)
# ----------------------------------------------------------------------


class _MacBackend(_Backend):
    """macOS backend using CoreGraphics for input + screenshot."""

    def __init__(self) -> None:
        try:
            import Quartz  # type: ignore
            import AppKit  # type: ignore
        except ImportError as exc:
            raise _DesktopError(
                "macOS desktop backend requires PyObjC. Install with: "
                "pip install 'pyobjc-framework-Quartz' 'pyobjc-framework-Cocoa' "
                "(or `pip install nanobot-ai[desktop]` when shipped)."
            ) from exc
        self.Quartz = Quartz
        self.AppKit = AppKit

    # -- screen / cursor info --------------------------------------------------

    def screen_size(self) -> tuple[int, int]:
        screen = self.AppKit.NSScreen.mainScreen()
        if screen is None:
            raise _DesktopError("no main screen available")
        frame = screen.frame()
        return int(frame.size.width), int(frame.size.height)

    def cursor_position(self) -> tuple[int, int]:
        loc = self.Quartz.CGEventGetLocation(self.Quartz.CGEventCreate(None))
        return int(loc.x), int(loc.y)

    def info(self) -> dict[str, Any]:
        w, h = self.screen_size()
        cx, cy = self.cursor_position()
        return {
            "platform": "darwin",
            "screen": {"width": w, "height": h},
            "cursor": {"x": cx, "y": cy},
        }

    # -- screenshot ------------------------------------------------------------

    def screenshot(self, path: str) -> None:
        Q = self.Quartz
        image = Q.CGWindowListCreateImage(
            Q.CGRectInfinite,
            Q.kCGWindowListOptionOnScreenOnly,
            Q.kCGNullWindowID,
            Q.kCGWindowImageDefault,
        )
        if image is None:
            raise _DesktopError(
                "CGWindowListCreateImage returned None. Grant Screen "
                "Recording permission in System Settings > Privacy & Security."
            )
        url = self._file_url(path)
        dest = Q.CGImageDestinationCreateWithURL(url, "public.png", 1, None)
        if dest is None:
            raise _DesktopError(f"failed to create image destination for {path}")
        Q.CGImageDestinationAddImage(dest, image, None)
        if not Q.CGImageDestinationFinalize(dest):
            raise _DesktopError(f"failed to write screenshot to {path}")

    def resize_png(self, path: str, width: int, height: int) -> None:
        Q = self.Quartz
        url = self._file_url(path)
        src = Q.CGImageSourceCreateWithURL(url, None)
        if src is None or Q.CGImageSourceGetCount(src) == 0:
            raise _DesktopError(f"cannot read PNG for resize: {path}")
        original = Q.CGImageSourceCreateImageAtIndex(src, 0, None)
        if original is None:
            raise _DesktopError(f"cannot decode PNG for resize: {path}")
        color_space = Q.CGImageGetColorSpace(original)
        ctx = Q.CGBitmapContextCreate(
            None, width, height, 8, 0, color_space,
            Q.kCGImageAlphaPremultipliedLast,
        )
        if ctx is None:
            raise _DesktopError(f"failed to create bitmap context for resize: {path}")
        rect = Q.CGRectMake(0, 0, width, height)
        Q.CGContextDrawImage(ctx, rect, original)
        scaled = Q.CGBitmapContextCreateImage(ctx)
        dest = Q.CGImageDestinationCreateWithURL(url, "public.png", 1, None)
        Q.CGImageDestinationAddImage(dest, scaled, None)
        if not Q.CGImageDestinationFinalize(dest):
            raise _DesktopError(f"failed to write resized PNG: {path}")

    def _file_url(self, path: str):
        return self.AppKit.NSURL.fileURLWithPath_(path)

    # -- mouse -----------------------------------------------------------------

    def mouse_move(self, x: int, y: int) -> None:
        Q = self.Quartz
        cx, cy = self.cursor_position()
        dx, dy = x - cx, y - cy
        dist = (dx * dx + dy * dy) ** 0.5
        if dist < 2:
            evt = Q.CGEventCreateMouseEvent(
                None, Q.kCGEventMouseMoved, (x, y), Q.kCGMouseButtonLeft
            )
            Q.CGEventPost(Q.kCGHIDEventTap, evt)
            return
        steps = max(5, min(40, int(dist / 8)))
        for i in range(1, steps + 1):
            ix = cx + dx * i / steps
            iy = cy + dy * i / steps
            evt = Q.CGEventCreateMouseEvent(
                None, Q.kCGEventMouseMoved, (ix, iy), Q.kCGMouseButtonLeft
            )
            Q.CGEventPost(Q.kCGHIDEventTap, evt)
            time.sleep(0.015)

    def mouse_click(self, x: int, y: int, button: str, clicks: int) -> None:
        Q = self.Quartz
        button_map = {
            "left": (Q.kCGEventLeftMouseDown, Q.kCGEventLeftMouseUp, Q.kCGMouseButtonLeft),
            "right": (Q.kCGEventRightMouseDown, Q.kCGEventRightMouseUp, Q.kCGMouseButtonRight),
            "middle": (Q.kCGEventOtherMouseDown, Q.kCGEventOtherMouseUp, Q.kCGMouseButtonCenter),
        }
        if button not in button_map:
            raise _DesktopError(f"unknown mouse button: {button}")
        down_t, up_t, btn = button_map[button]
        for i in range(clicks):
            down = Q.CGEventCreateMouseEvent(None, down_t, (x, y), btn)
            up = Q.CGEventCreateMouseEvent(None, up_t, (x, y), btn)
            Q.CGEventSetIntegerValueField(down, Q.kCGMouseEventClickState, i + 1)
            Q.CGEventSetIntegerValueField(up, Q.kCGMouseEventClickState, i + 1)
            Q.CGEventPost(Q.kCGHIDEventTap, down)
            Q.CGEventPost(Q.kCGHIDEventTap, up)
            if clicks > 1 and i < clicks - 1:
                time.sleep(0.05)

    def mouse_drag(self, x1: int, y1: int, x2: int, y2: int) -> None:
        Q = self.Quartz
        btn = Q.kCGMouseButtonLeft
        down = Q.CGEventCreateMouseEvent(None, Q.kCGEventLeftMouseDown, (x1, y1), btn)
        Q.CGEventPost(Q.kCGHIDEventTap, down)
        time.sleep(0.05)
        # Animate to make the move visible.
        steps = 20
        for i in range(1, steps + 1):
            ix = x1 + (x2 - x1) * i / steps
            iy = y1 + (y2 - y1) * i / steps
            drag = Q.CGEventCreateMouseEvent(
                None, Q.kCGEventLeftMouseDragged, (ix, iy), btn
            )
            Q.CGEventPost(Q.kCGHIDEventTap, drag)
            time.sleep(0.01)
        up = Q.CGEventCreateMouseEvent(None, Q.kCGEventLeftMouseUp, (x2, y2), btn)
        Q.CGEventPost(Q.kCGHIDEventTap, up)

    def scroll(self, direction: str, amount: int) -> None:
        Q = self.Quartz
        dy = dx = 0
        magnitude = amount * 3  # ~3 pixel lines per notch
        if direction == "up":
            dy = magnitude
        elif direction == "down":
            dy = -magnitude
        elif direction == "left":
            dx = -magnitude
        elif direction == "right":
            dx = magnitude
        evt = Q.CGEventCreateScrollWheelEvent(
            None, Q.kCGScrollEventUnitLine, 2, dy, dx
        )
        Q.CGEventPost(Q.kCGHIDEventTap, evt)

    # -- keyboard --------------------------------------------------------------

    def type_text(self, text: str, delay_ms: int, delays_ms: list[int] | None = None) -> None:
        Q = self.Quartz
        # When per-character delays are supplied (humanize mode), use them
        # in order; otherwise fall back to the uniform fixed delay.
        if delays_ms:
            delays_iter = iter(delays_ms)
        else:
            delays_iter = None
        for ch in text:
            evt_down = Q.CGEventCreateKeyboardEvent(None, 0, True)
            evt_up = Q.CGEventCreateKeyboardEvent(None, 0, False)
            # Unicode injection: bypass keycode mapping. Each event carries
            # the literal UTF-16 char; this is how Anthropic's xdotool path
            # ultimately gets translated on Linux too.
            buf = ch.encode("utf-16-le")
            Q.CGEventKeyboardSetUnicodeString(evt_down, len(buf) // 2, ch)
            Q.CGEventKeyboardSetUnicodeString(evt_up, len(buf) // 2, ch)
            Q.CGEventPost(Q.kCGHIDEventTap, evt_down)
            Q.CGEventPost(Q.kCGHIDEventTap, evt_up)
            # Pick the delay for this keystroke. With humanize mode the
            # generator may yield MORE entries than chars (each entry is
            # "the gap after position k"); the first entry is the pre-first
            # keystroke startup latency, so we sleep *before* the next char.
            cur: int = 0
            if delays_iter is not None:
                try:
                    cur = int(next(delays_iter))
                except StopIteration:
                    cur = delay_ms
            else:
                cur = delay_ms
            if cur > 0:
                time.sleep(cur / 1000.0)

    def key_chord(self, chord: str) -> None:
        Q = self.Quartz
        keycode, flags, modifier_only = self._resolve_chord(chord)
        if modifier_only:
            # Tap a modifier on its own: brief down -> up of the modifier's
            # physical keycode. The flags themselves are implicit.
            down = Q.CGEventCreateKeyboardEvent(None, keycode, True)
            up = Q.CGEventCreateKeyboardEvent(None, keycode, False)
            Q.CGEventPost(Q.kCGHIDEventTap, down)
            Q.CGEventPost(Q.kCGHIDEventTap, up)
            return
        mod_keycodes = self._chord_modifier_keycodes(chord)
        for mk in mod_keycodes:
            mod_down = Q.CGEventCreateKeyboardEvent(None, mk, True)
            Q.CGEventPost(Q.kCGHIDEventTap, mod_down)
        down = Q.CGEventCreateKeyboardEvent(None, keycode, True)
        up = Q.CGEventCreateKeyboardEvent(None, keycode, False)
        if flags:
            Q.CGEventSetFlags(down, flags)
            Q.CGEventSetFlags(up, flags)
        Q.CGEventPost(Q.kCGHIDEventTap, down)
        Q.CGEventPost(Q.kCGHIDEventTap, up)
        for mk in reversed(mod_keycodes):
            mod_up = Q.CGEventCreateKeyboardEvent(None, mk, False)
            Q.CGEventPost(Q.kCGHIDEventTap, mod_up)

    def hold_key(self, chord: str, duration: float) -> None:
        Q = self.Quartz
        keycode, flags, modifier_only = self._resolve_chord(chord)
        if modifier_only:
            down = Q.CGEventCreateKeyboardEvent(None, keycode, True)
            up = Q.CGEventCreateKeyboardEvent(None, keycode, False)
            Q.CGEventPost(Q.kCGHIDEventTap, down)
            time.sleep(duration)
            Q.CGEventPost(Q.kCGHIDEventTap, up)
            return
        mod_keycodes = self._chord_modifier_keycodes(chord)
        for mk in mod_keycodes:
            mod_down = Q.CGEventCreateKeyboardEvent(None, mk, True)
            Q.CGEventPost(Q.kCGHIDEventTap, mod_down)
        down = Q.CGEventCreateKeyboardEvent(None, keycode, True)
        up = Q.CGEventCreateKeyboardEvent(None, keycode, False)
        if flags:
            Q.CGEventSetFlags(down, flags)
            Q.CGEventSetFlags(up, flags)
        Q.CGEventPost(Q.kCGHIDEventTap, down)
        time.sleep(duration)
        Q.CGEventPost(Q.kCGHIDEventTap, up)
        for mk in reversed(mod_keycodes):
            mod_up = Q.CGEventCreateKeyboardEvent(None, mk, False)
            Q.CGEventPost(Q.kCGHIDEventTap, mod_up)

    def _resolve_chord(self, chord: str) -> tuple[int, int, bool]:
        """Parse 'Cmd+Shift+S' / 'Return' / 'Shift' into (keycode, flags, modifier_only).

        ``modifier_only`` is True when the chord contains *only* modifier
        names (e.g. ``"Shift"`` or ``"Cmd+Option"``). In that case the
        returned ``keycode`` is the physical key for the last-named modifier
        and ``flags`` is 0 — callers should not apply flags on top, because
        macOS derives the flag bit from the physical key event itself.
        """
        Q = self.Quartz
        parts = [p.strip() for p in chord.split("+") if p.strip()]
        if not parts:
            raise _DesktopError(f"empty key chord: {chord!r}")
        mods = 0
        key_name: str | None = None
        mod_map = {
            "cmd": Q.kCGEventFlagMaskCommand, "command": Q.kCGEventFlagMaskCommand,
            "shift": Q.kCGEventFlagMaskShift,
            "alt": Q.kCGEventFlagMaskAlternate, "option": Q.kCGEventFlagMaskAlternate,
            "opt": Q.kCGEventFlagMaskAlternate,
            "ctrl": Q.kCGEventFlagMaskControl, "control": Q.kCGEventFlagMaskControl,
            "fn": Q.kCGEventFlagMaskSecondaryFn,
        }
        # Physical virtual keycodes for the modifier keys themselves
        # (kVK_Shift, kVK_Command, kVK_Option, kVK_Control, kVK_Function).
        mod_keycode = {
            "shift": 0x38,
            "cmd": 0x37, "command": 0x37,
            "alt": 0x3A, "option": 0x3A, "opt": 0x3A,
            "ctrl": 0x3B, "control": 0x3B,
            "fn": 0x3F,
        }
        last_mod: str | None = None
        for part in parts:
            lower = part.lower()
            if lower in mod_map:
                mods |= mod_map[lower]
                last_mod = lower
            else:
                if key_name is not None:
                    raise _DesktopError(
                        f"chord has multiple non-modifier keys: {chord!r}"
                    )
                key_name = part
        if key_name is None:
            # Modifier-only chord: emit the physical modifier key.
            assert last_mod is not None  # parts non-empty => at least one mod
            return mod_keycode[last_mod], 0, True
        return self._keycode_for(key_name), mods, False

    def _chord_modifier_keycodes(self, chord: str) -> list[int]:
        """Return ordered physical keycodes for each modifier in *chord*."""
        parts = [p.strip() for p in chord.split("+") if p.strip()]
        keycodes: list[int] = []
        mod_names = {"cmd", "command", "shift", "alt", "option", "opt", "ctrl", "control", "fn"}
        for part in parts:
            if part.lower() in mod_names:
                keycodes.append(self._chord_mod_keycode(part))
        return keycodes

    def _chord_mod_keycode(self, name: str) -> int:
        lower = name.lower()
        if lower in ("cmd", "command"):
            return 0x37
        if lower == "shift":
            return 0x38
        if lower in ("alt", "option", "opt"):
            return 0x3A
        if lower in ("ctrl", "control"):
            return 0x3B
        if lower == "fn":
            return 0x3F
        return 0x37  # default to Cmd

    # ANSI keyboard virtual keycodes. Mirrors NSEvent.h kVK_* constants.
    _VK = {
        "a": 0x00, "s": 0x01, "d": 0x02, "f": 0x03, "h": 0x04, "g": 0x05,
        "z": 0x06, "x": 0x07, "c": 0x08, "v": 0x09, "b": 0x0B, "q": 0x0C,
        "w": 0x0D, "e": 0x0E, "r": 0x0F, "y": 0x10, "t": 0x11,
        "1": 0x12, "2": 0x13, "3": 0x14, "4": 0x15, "6": 0x16, "5": 0x17,
        "=": 0x18, "9": 0x19, "7": 0x1A, "-": 0x1B, "8": 0x1C, "0": 0x1D,
        "]": 0x1E, "o": 0x1F, "u": 0x20, "[": 0x21, "i": 0x22, "p": 0x23,
        "l": 0x25, "j": 0x26, "'": 0x27, "k": 0x28, ";": 0x29, "\\": 0x2A,
        ",": 0x2B, "/": 0x2C, "n": 0x2D, "m": 0x2E, ".": 0x2F, "`": 0x32,
        "return": 0x24, "enter": 0x24,
        "tab": 0x30, "space": 0x31, "backspace": 0x33, "delete": 0x33,
        "escape": 0x35, "esc": 0x35,
        "left": 0x7B, "right": 0x7C, "down": 0x7D, "up": 0x7E,
        "home": 0x73, "end": 0x77, "pageup": 0x74, "pagedown": 0x79,
        "f1": 0x7A, "f2": 0x78, "f3": 0x63, "f4": 0x76, "f5": 0x60,
        "f6": 0x61, "f7": 0x62, "f8": 0x64, "f9": 0x65, "f10": 0x6D,
        "f11": 0x67, "f12": 0x6F,
    }

    def _keycode_for(self, name: str) -> int:
        key = name.lower()
        if key in self._VK:
            return self._VK[key]
        if key in _KEY_ALIASES:
            return self._VK[_KEY_ALIASES[key].lower()]
        if len(key) == 1 and key.isalnum():
            # Single char not in table -> let the unicode path handle it via
            # a synthetic event with no keycode.
            return 0
        raise _DesktopError(f"unknown key: {name!r}")
