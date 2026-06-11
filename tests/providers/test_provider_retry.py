import asyncio

import pytest

from nanobot.providers.base import GenerationSettings, LLMProvider, LLMResponse


class ScriptedProvider(LLMProvider):
    def __init__(self, responses):
        super().__init__()
        self._responses = list(responses)
        self.calls = 0
        self.last_kwargs: dict = {}

    async def chat(self, *args, **kwargs) -> LLMResponse:
        self.calls += 1
        self.last_kwargs = kwargs
        response = self._responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    def get_default_model(self) -> str:
        return "test-model"


@pytest.mark.asyncio
async def test_chat_with_retry_retries_transient_error_then_succeeds(monkeypatch) -> None:
    provider = ScriptedProvider([
        LLMResponse(content="429 rate limit", finish_reason="error"),
        LLMResponse(content="ok"),
    ])
    delays: list[int] = []

    async def _fake_sleep(delay: int) -> None:
        delays.append(delay)

    monkeypatch.setattr("nanobot.providers.base.asyncio.sleep", _fake_sleep)

    response = await provider.chat_with_retry(messages=[{"role": "user", "content": "hello"}])

    assert response.finish_reason == "stop"
    assert response.content == "ok"
    assert provider.calls == 2
    assert delays == [1]


@pytest.mark.asyncio
async def test_chat_with_retry_does_not_retry_non_transient_error(monkeypatch) -> None:
    provider = ScriptedProvider([
        LLMResponse(content="401 unauthorized", finish_reason="error"),
    ])
    delays: list[int] = []

    async def _fake_sleep(delay: int) -> None:
        delays.append(delay)

    monkeypatch.setattr("nanobot.providers.base.asyncio.sleep", _fake_sleep)

    response = await provider.chat_with_retry(messages=[{"role": "user", "content": "hello"}])

    assert response.content == "401 unauthorized"
    assert provider.calls == 1
    assert delays == []


@pytest.mark.asyncio
async def test_chat_with_retry_returns_final_error_after_retries(monkeypatch) -> None:
    provider = ScriptedProvider([
        LLMResponse(content="429 rate limit a", finish_reason="error"),
        LLMResponse(content="429 rate limit b", finish_reason="error"),
        LLMResponse(content="429 rate limit c", finish_reason="error"),
        LLMResponse(content="503 final server error", finish_reason="error"),
    ])
    delays: list[int] = []

    async def _fake_sleep(delay: int) -> None:
        delays.append(delay)

    monkeypatch.setattr("nanobot.providers.base.asyncio.sleep", _fake_sleep)

    response = await provider.chat_with_retry(messages=[{"role": "user", "content": "hello"}])

    assert response.content == "503 final server error"
    assert provider.calls == 4
    assert delays == [1, 2, 4]


@pytest.mark.asyncio
async def test_chat_with_retry_does_not_restart_after_key_pool_exhausted(monkeypatch) -> None:
    provider = ScriptedProvider([
        LLMResponse(
            content="Error: All configured API keys were rate-limited or out of quota after trying 5 keys. Last error: 429 quota exceeded",
            finish_reason="error",
        ),
    ])
    delays: list[int] = []

    async def _fake_sleep(delay: int) -> None:
        delays.append(delay)

    monkeypatch.setattr("nanobot.providers.base.asyncio.sleep", _fake_sleep)

    response = await provider.chat_with_retry(messages=[{"role": "user", "content": "hello"}])

    assert response.finish_reason == "error"
    assert provider.calls == 1
    assert delays == []


@pytest.mark.asyncio
async def test_chat_stream_with_retry_does_not_restart_after_key_pool_exhausted(monkeypatch) -> None:
    provider = ScriptedProvider([
        LLMResponse(
            content="Error: All configured API keys were rate-limited or out of quota after trying 5 keys. Last error: 429 quota exceeded",
            finish_reason="error",
        ),
    ])
    delays: list[int] = []

    async def _fake_sleep(delay: int) -> None:
        delays.append(delay)

    monkeypatch.setattr("nanobot.providers.base.asyncio.sleep", _fake_sleep)

    response = await provider.chat_stream_with_retry(messages=[{"role": "user", "content": "hello"}])

    assert response.finish_reason == "error"
    assert provider.calls == 1
    assert delays == []


@pytest.mark.asyncio
async def test_chat_with_retry_preserves_cancelled_error() -> None:
    provider = ScriptedProvider([asyncio.CancelledError()])

    with pytest.raises(asyncio.CancelledError):
        await provider.chat_with_retry(messages=[{"role": "user", "content": "hello"}])


@pytest.mark.asyncio
async def test_chat_with_retry_uses_provider_generation_defaults() -> None:
    """When callers omit generation params, provider.generation defaults are used."""
    provider = ScriptedProvider([LLMResponse(content="ok")])
    provider.generation = GenerationSettings(temperature=0.2, max_tokens=321, reasoning_effort="high")

    await provider.chat_with_retry(messages=[{"role": "user", "content": "hello"}])

    assert provider.last_kwargs["temperature"] == 0.2
    assert provider.last_kwargs["max_tokens"] == 321
    assert provider.last_kwargs["reasoning_effort"] == "high"


@pytest.mark.asyncio
async def test_chat_with_retry_explicit_override_beats_defaults() -> None:
    """Explicit kwargs should override provider.generation defaults."""
    provider = ScriptedProvider([LLMResponse(content="ok")])
    provider.generation = GenerationSettings(temperature=0.2, max_tokens=321, reasoning_effort="high")

    await provider.chat_with_retry(
        messages=[{"role": "user", "content": "hello"}],
        temperature=0.9,
        max_tokens=9999,
        reasoning_effort="low",
    )

    assert provider.last_kwargs["temperature"] == 0.9
    assert provider.last_kwargs["max_tokens"] == 9999
    assert provider.last_kwargs["reasoning_effort"] == "low"


# ---------------------------------------------------------------------------
# Image fallback tests
# ---------------------------------------------------------------------------

_IMAGE_MSG = [
    {"role": "user", "content": [
        {"type": "text", "text": "describe this"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}, "_meta": {"path": "/media/test.png"}},
    ]},
]

_IMAGE_MSG_NO_META = [
    {"role": "user", "content": [
        {"type": "text", "text": "describe this"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
    ]},
]


@pytest.mark.asyncio
async def test_non_transient_error_with_images_retries_without_images() -> None:
    """Any non-transient error retries once with images stripped when images are present."""
    provider = ScriptedProvider([
        LLMResponse(content="API调用参数有误,请检查文档", finish_reason="error"),
        LLMResponse(content="ok, no image"),
    ])

    response = await provider.chat_with_retry(messages=_IMAGE_MSG)

    assert response.content == "ok, no image"
    assert provider.calls == 2
    msgs_on_retry = provider.last_kwargs["messages"]
    for msg in msgs_on_retry:
        content = msg.get("content")
        if isinstance(content, list):
            assert all(b.get("type") != "image_url" for b in content)
            assert any(
                "IMAGE DROPPED" in (b.get("text") or "")
                and "/media/test.png" in (b.get("text") or "")
                for b in content
            )


@pytest.mark.asyncio
async def test_non_transient_error_without_images_no_retry() -> None:
    """Non-transient errors without image content are returned immediately."""
    provider = ScriptedProvider([
        LLMResponse(content="401 unauthorized", finish_reason="error"),
    ])

    response = await provider.chat_with_retry(
        messages=[{"role": "user", "content": "hello"}],
    )

    assert provider.calls == 1
    assert response.finish_reason == "error"


@pytest.mark.asyncio
async def test_image_fallback_returns_error_on_second_failure() -> None:
    """If the image-stripped retry also fails, return that error."""
    provider = ScriptedProvider([
        LLMResponse(content="some model error", finish_reason="error"),
        LLMResponse(content="still failing", finish_reason="error"),
    ])

    response = await provider.chat_with_retry(messages=_IMAGE_MSG)

    assert provider.calls == 2
    assert response.content == "still failing"
    assert response.finish_reason == "error"


@pytest.mark.asyncio
async def test_image_fallback_without_meta_uses_default_placeholder() -> None:
    """When _meta is absent, fallback placeholder warns the agent it cannot see the image."""
    provider = ScriptedProvider([
        LLMResponse(content="error", finish_reason="error"),
        LLMResponse(content="ok"),
    ])

    response = await provider.chat_with_retry(messages=_IMAGE_MSG_NO_META)

    assert response.content == "ok"
    assert provider.calls == 2
    msgs_on_retry = provider.last_kwargs["messages"]
    for msg in msgs_on_retry:
        content = msg.get("content")
        if isinstance(content, list):
            assert any("IMAGE DROPPED" in (b.get("text") or "") for b in content)


@pytest.mark.asyncio
async def test_payload_too_large_error_uses_payload_specific_warning() -> None:
    """413/payload-too-large errors produce a payload-specific placeholder, not vision-rejection text."""
    provider = ScriptedProvider([
        LLMResponse(content="Error: Request Entity Too Large", finish_reason="error"),
        LLMResponse(content="ok"),
    ])

    response = await provider.chat_with_retry(messages=_IMAGE_MSG)

    assert response.content == "ok"
    assert provider.calls == 2
    msgs_on_retry = provider.last_kwargs["messages"]
    payload_warning_found = False
    for msg in msgs_on_retry:
        content = msg.get("content")
        if isinstance(content, list):
            for b in content:
                text = b.get("text") or ""
                if "IMAGE DROPPED" in text and "payload" in text.lower():
                    payload_warning_found = True
    assert payload_warning_found, "expected payload-too-large specific warning in placeholder"


def _image_block(path: str) -> dict:
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,abc-{path}"},
        "_meta": {"path": path},
    }


@pytest.mark.asyncio
async def test_prune_old_images_keeps_only_most_recent() -> None:
    """Sliding window keeps the most-recent image and replaces older ones with a placeholder."""
    provider = ScriptedProvider([LLMResponse(content="ok")])
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "look"}, _image_block("/a.png")]},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": [{"type": "text", "text": "now this"}, _image_block("/b.png")]},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": [{"type": "text", "text": "and this"}, _image_block("/c.png")]},
    ]

    response = await provider.chat_with_retry(messages=messages)

    assert response.content == "ok"
    sent = provider.last_kwargs["messages"]

    image_blocks = []
    placeholders = []
    for msg in sent:
        content = msg.get("content")
        if isinstance(content, list):
            for b in content:
                if isinstance(b, dict) and b.get("type") == "image_url":
                    image_blocks.append(b)
                elif isinstance(b, dict) and b.get("type") == "text":
                    text = b.get("text") or ""
                    if "earlier screenshot omitted" in text:
                        placeholders.append(text)

    # Default _IMAGE_HISTORY_KEEP is 1 → only newest image survives.
    assert len(image_blocks) == 1
    assert image_blocks[0]["_meta"]["path"] == "/c.png"
    assert len(placeholders) == 2
    assert any("/a.png" in p for p in placeholders)
    assert any("/b.png" in p for p in placeholders)


@pytest.mark.asyncio
async def test_prune_old_images_noop_when_under_budget() -> None:
    """When image count is within budget, messages are passed through unchanged."""
    provider = ScriptedProvider([LLMResponse(content="ok")])
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "one"}, _image_block("/only.png")]},
    ]

    response = await provider.chat_with_retry(messages=messages)

    assert response.content == "ok"
    sent = provider.last_kwargs["messages"]
    images = [
        b for msg in sent
        if isinstance(msg.get("content"), list)
        for b in msg["content"]
        if isinstance(b, dict) and b.get("type") == "image_url"
    ]
    assert len(images) == 1
    assert images[0]["_meta"]["path"] == "/only.png"
