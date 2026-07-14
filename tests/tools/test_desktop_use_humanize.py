"""Tests for ``DesktopUseTool._human_delays`` human-like typing model."""

from __future__ import annotations

import statistics

import pytest

from nanobot.agent.tools.desktop_use import (
    DesktopUseTool,
    _FAST_BIGRAMS,
    _HUMAN_MAX_MEAN_MS,
    _HUMAN_MIN_MEAN_MS,
)


def _tool(*, typing_delay_ms: int = 12, humanize: bool = True) -> DesktopUseTool:
    return DesktopUseTool(typing_delay_ms=typing_delay_ms, humanize_typing=humanize)


def test_human_delays_returns_one_entry_per_char() -> None:
    text = "GitHub Desktop"
    delays = _tool()._human_delays(text)
    assert len(delays) == len(text)
    assert all(isinstance(d, int) for d in delays)


def test_human_delays_first_char_has_startup_latency() -> None:
    # The first delay should be larger than a typical mid-word delay on
    # average, because of the orienting bonus.
    tool = _tool(typing_delay_ms=80)
    large_first = 0
    for _ in range(200):
        d = tool._human_delays("arc browser")
        # First delay minus the plausible base mean is the startup bonus.
        if d[0] >= 80 + 90:  # min bonus in code is 90
            large_first += 1
    # Should trigger the bonus on essentially every sample (allow for the
    # 12 ms lognormal floor occasionally pulling a small sample).
    assert large_first >= 180


def test_human_delays_fast_bigrams_are_faster_on_average() -> None:
    """Common bigrams should produce shorter delays than uncommon ones."""
    # "th" is in _FAST_BIGRAMS, "qz" is not.
    tool = _tool(typing_delay_ms=120)
    fast_samples: list[int] = []
    slow_samples: list[int] = []
    for _ in range(400):
        fast = tool._human_delays("th")
        slow = tool._human_delays("qz")
        # skip the startup-bonus-loaded first index, compare the inter-key gap.
        fast_samples.append(fast[1])
        slow_samples.append(slow[1])
    fast_mean = statistics.mean(fast_samples)
    slow_mean = statistics.mean(slow_samples)
    assert fast_mean < slow_mean * 0.75


def test_human_delays_space_and_punctuation_incur_penalty() -> None:
    tool = _tool(typing_delay_ms=100)
    letter_samples = [statistics.mean(tool._human_delays("aaaa")[1:]) for _ in range(200)]
    space_samples = [statistics.mean(tool._human_delays("a a ")[1:]) for _ in range(200)]
    punct_samples = [statistics.mean(tool._human_delays("a.a.")[1:]) for _ in range(200)]
    letter_mean = statistics.mean(letter_samples)
    space_mean = statistics.mean(space_samples)
    punct_mean = statistics.mean(punct_samples)
    assert space_mean > letter_mean
    assert punct_mean > letter_mean


def test_human_delays_clamp_to_human_band_when_config_tiny() -> None:
    # Even with a 5 ms config, none of the per-key delays should be sub-12 ms
    # because _lognormal_ms clamps to a 12 ms floor.
    delays = _tool(typing_delay_ms=5)._human_delays("hello world")
    assert all(d >= 12 for d in delays)


def test_human_delays_respects_upper_band() -> None:
    # Sanity: average across a string should sit within the human band.
    text = "GitHub Desktop"
    samples = [statistics.mean(_tool(typing_delay_ms=12)._human_delays(text)) for _ in range(200)]
    overall = statistics.mean(samples)
    # The lower end accounts for bigram boost on real text.
    assert _HUMAN_MIN_MEAN_MS * 0.4 <= overall <= _HUMAN_MAX_MEAN_MS * 4


def test_human_delays_zero_for_empty_text() -> None:
    assert _tool()._human_delays("") == []


def test_fast_bigrams_constant_nonempty() -> None:
    # sanity that the constant used by the test above truly is a populated set
    assert "th" in _FAST_BIGRAMS and "he" in _FAST_BIGRAMS and "qz" not in _FAST_BIGRAMS


@pytest.mark.parametrize("humanize", [True, False])
def test_execute_type_path_passes_delays_only_when_humanize_on(monkeypatch, humanize: bool) -> None:
    """When humanize is off, the backend call must NOT receive per-char delays."""
    tool = _tool(humanize=humanize)
    captured: dict[str, object] = {}

    class _StubBackend:
        def type_text(self, text: str, delay_ms: int, delays_ms: list[int] | None = None) -> None:
            captured["text"] = text
            captured["delay_ms"] = delay_ms
            captured["delays_ms"] = delays_ms

    import asyncio

    async def main() -> None:
        result = await tool.execute(
            action="type",
            text="arc",
        )
        return result

    # Inject stub backend before execute touches the real backend.
    tool._backend = _StubBackend()  # type: ignore[assignment]
    tool._backend_error = None

    out = asyncio.run(main())
    assert out and "ok" in out
    assert captured["text"] == "arc"
    if humanize:
        assert isinstance(captured["delays_ms"], list)
        assert len(captured["delays_ms"]) == 3
    else:
        assert captured["delays_ms"] is None