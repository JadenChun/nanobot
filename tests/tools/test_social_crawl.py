import base64
import io
import json
import socket
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.tools.social_crawl import SocialCrawlClient, SocialCrawlTool
from nanobot.crawler.worker import BrowserSession, Crawl4AIWorker, WorkerRequestError


def _result(
    html: str,
    url: str = "https://www.instagram.com/p/example/",
    media: dict | None = None,
    screenshot: str | None = None,
    js_execution_result: dict | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        cleaned_html=html,
        url=url,
        status_code=200,
        media=media or {},
        screenshot=screenshot,
        js_execution_result=js_execution_result,
    )


def test_worker_returns_html_without_replacing_it_with_extracted_fields() -> None:
    html = (
        '<article class="post"><a href="/cats">Cat owners</a>'
        "<p>Discussion about tofu litter</p></article>"
    )

    payload = Crawl4AIWorker._html_payload(
        _result(html), session_id="session-1", offset=0, max_chars=30000
    )

    assert payload["html"] == html
    assert payload["html_total_chars"] == len(html)
    assert payload["html_truncated"] is False
    assert "findings" not in payload
    assert "summary" not in payload


def test_worker_paginates_large_html_without_summarizing() -> None:
    html = "<main>" + ("cat owner comment " * 1000) + "</main>"

    first = Crawl4AIWorker._html_payload(
        _result(html), session_id="session-1", offset=0, max_chars=1000
    )
    second = Crawl4AIWorker._html_payload(
        _result(html),
        session_id="session-1",
        offset=first["next_html_offset"],
        max_chars=1000,
    )

    assert first["html"] == html[:1000]
    assert second["html"] == html[1000:2000]
    assert first["html_truncated"] is True


def test_worker_includes_image_metadata_in_html_evidence() -> None:
    result = _result(
        "<main>Post caption</main>",
        media={
            "images": [
                {
                    "src": "https://static.fbcdn.net/cat.jpg",
                    "alt": "Cat beside a tofu litter bag",
                    "desc": "Promotional image for cat owners",
                }
            ]
        },
    )

    payload = Crawl4AIWorker._html_payload(
        result,
        session_id="session-1",
        offset=0,
        max_chars=30000,
    )

    assert 'data-crawl4ai-media="images"' in payload["html"]
    assert 'src="https://static.fbcdn.net/cat.jpg"' in payload["html"]
    assert 'alt="Cat beside a tofu litter bag"' in payload["html"]
    assert 'data-description="Promotional image for cat owners"' in payload["html"]


def test_worker_prefers_visible_viewport_html_over_full_document() -> None:
    viewport = (
        '<section data-crawl4ai-viewport="true"><span>Visible post text</span></section>'
    )
    result = _result(
        "<main>Huge document header</main>",
        js_execution_result={"success": True, "results": [None, viewport]},
    )

    payload = Crawl4AIWorker._html_payload(
        result, session_id="session-1", offset=0, max_chars=12000
    )

    assert payload["html"] == viewport
    assert payload["content_scope"] == "visible_viewport"


def test_worker_returns_a_bounded_visual_screenshot() -> None:
    from PIL import Image

    source = io.BytesIO()
    Image.new("RGB", (1600, 900), "navy").save(source, format="PNG")
    result = _result(
        "<main>Visible post</main>",
        screenshot=base64.b64encode(source.getvalue()).decode("ascii"),
    )

    payload = Crawl4AIWorker._html_payload(
        result, session_id="session-1", offset=0, max_chars=12000
    )

    assert payload["screenshot_mime"] == "image/jpeg"
    screenshot = base64.b64decode(payload["screenshot_base64"])
    assert len(screenshot) <= 180 * 1024
    with Image.open(io.BytesIO(screenshot)) as rendered:
        assert rendered.width <= 1280
        assert rendered.height <= 1280


def test_worker_url_policy_requires_https_and_an_allowlisted_domain() -> None:
    worker = Crawl4AIWorker("worker.sock", {"instagram.com"})

    assert worker._validate_url("https://www.instagram.com/explore/tags/catfood/")
    with pytest.raises(WorkerRequestError, match="HTTPS"):
        worker._validate_url("http://www.instagram.com/explore/tags/catfood/")
    with pytest.raises(WorkerRequestError, match="not allowlisted"):
        worker._validate_url("https://example.com/")
    with pytest.raises(WorkerRequestError, match="IP-literal"):
        worker._validate_url("https://127.0.0.1/")


def test_browser_route_filter_blocks_non_allowlisted_redirect_targets() -> None:
    worker = Crawl4AIWorker("worker.sock", {"instagram.com", "cdninstagram.com"})

    assert worker._route_url_allowed("https://www.instagram.com/p/example/") is True
    assert worker._route_url_allowed("https://static.cdninstagram.com/script.js") is True
    assert worker._route_url_allowed("https://169.254.169.254/latest/meta-data/") is False
    assert worker._route_url_allowed("https://example.com/redirect") is False
    assert worker._route_url_allowed("file:///etc/passwd") is False


@pytest.mark.asyncio
@pytest.mark.parametrize("resource_type", ["image", "media", "font", "stylesheet", "script"])
async def test_browser_route_filter_allows_normal_page_resources(resource_type) -> None:
    worker = Crawl4AIWorker("worker.sock", {"facebook.com", "fbcdn.net"})
    route = MagicMock()
    route.request.resource_type = resource_type
    route.request.url = "https://static.fbcdn.net/asset"
    route.continue_ = AsyncMock()
    route.abort = AsyncMock()
    context = MagicMock()

    async def install(_pattern, handler) -> None:
        await handler(route)

    context.route = AsyncMock(side_effect=install)

    await worker._install_route_filter(MagicMock(), context)

    route.continue_.assert_awaited_once()
    route.abort.assert_not_awaited()


@pytest.mark.asyncio
async def test_browser_route_filter_still_blocks_off_allowlist_resources() -> None:
    worker = Crawl4AIWorker("worker.sock", {"facebook.com", "fbcdn.net"})
    route = MagicMock()
    route.request.resource_type = "image"
    route.request.url = "https://untrusted.example/asset.jpg"
    route.continue_ = AsyncMock()
    route.abort = AsyncMock()
    context = MagicMock()

    async def install(_pattern, handler) -> None:
        await handler(route)

    context.route = AsyncMock(side_effect=install)

    await worker._install_route_filter(MagicMock(), context)

    route.abort.assert_awaited_once()
    route.continue_.assert_not_awaited()


def test_worker_tcp_transport_is_loopback_only() -> None:
    worker = Crawl4AIWorker(
        "unused.sock",
        {"egopetfood.com"},
        tcp_host="127.0.0.1",
        tcp_port=18791,
    )
    assert worker.tcp_host == "127.0.0.1"
    with pytest.raises(ValueError, match="loopback-only"):
        Crawl4AIWorker("unused.sock", {"egopetfood.com"}, tcp_host="0.0.0.0")


def test_social_crawl_client_can_use_loopback_tcp() -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]

    def respond() -> None:
        connection, _ = server.accept()
        with connection:
            request = json.loads(connection.recv(4096).decode("utf-8"))
            assert request == {"op": "health"}
            connection.sendall(b'{"ok":true,"status":"ready"}\n')
        server.close()

    thread = threading.Thread(target=respond, daemon=True)
    thread.start()
    response = SocialCrawlClient(tcp_host="127.0.0.1", tcp_port=port).request("health")
    thread.join(timeout=2)

    assert response == {"ok": True, "status": "ready"}


@pytest.mark.asyncio
async def test_click_action_is_built_server_side_and_returns_rendered_html() -> None:
    worker = Crawl4AIWorker("worker.sock", {"threads.net"})
    worker.sessions["session-1"] = BrowserSession(
        url="https://www.threads.net/t/example", updated_at=0
    )
    worker._expire_sessions = AsyncMock()
    worker._browser_context_alive = MagicMock(return_value=True)
    worker._validate_public_dns = AsyncMock()
    worker._crawl_result = AsyncMock(
        return_value=_result(
            '<div role="article"><p>Rendered reply</p></div>',
            "https://www.threads.net/t/example",
        )
    )

    response = await worker.dispatch(
        {
            "op": "click",
            "session_id": "session-1",
            "selector": "button[aria-label=Replies]",
            "wait_ms": 250,
        }
    )

    assert response["html"] == '<div role="article"><p>Rendered reply</p></div>'
    call = worker._crawl_result.await_args.kwargs
    assert call["js_only"] is True
    assert "document.querySelector" in call["js_code"]
    assert "button[aria-label=Replies]" in call["js_code"]


@pytest.mark.asyncio
async def test_navigate_reuses_the_existing_browser_session() -> None:
    worker = Crawl4AIWorker("worker.sock", {"facebook.com", "instagram.com"})
    worker.sessions["session-1"] = BrowserSession(
        url="https://www.facebook.com/example", updated_at=0
    )
    worker._expire_sessions = AsyncMock()
    worker._browser_context_alive = MagicMock(return_value=True)
    worker._validate_public_dns = AsyncMock()
    worker._crawl_result = AsyncMock(
        return_value=_result(
            "<main>Instagram page</main>",
            "https://www.instagram.com/example/",
        )
    )

    response = await worker.dispatch(
        {
            "op": "navigate",
            "session_id": "session-1",
            "url": "https://www.instagram.com/example/",
        }
    )

    assert response["session_id"] == "session-1"
    assert worker.sessions["session-1"].url == "https://www.instagram.com/example/"
    call = worker._crawl_result.await_args.kwargs
    assert call["session_id"] == "session-1"
    assert call["js_only"] is False
    assert call["screenshot"] is True


@pytest.mark.asyncio
async def test_authenticated_profile_disables_click_actions(tmp_path) -> None:
    worker = Crawl4AIWorker(
        "worker.sock",
        {"threads.net"},
        user_data_dir=str(tmp_path),
    )
    worker.sessions["session-1"] = BrowserSession(
        url="https://www.threads.net/t/example", updated_at=0
    )
    worker._expire_sessions = AsyncMock()
    worker._browser_context_alive = MagicMock(return_value=True)

    with pytest.raises(WorkerRequestError, match="click is disabled"):
        await worker.dispatch(
            {
                "op": "click",
                "session_id": "session-1",
                "selector": "button[aria-label=Like]",
            }
        )


@pytest.mark.asyncio
async def test_worker_health_reports_profile_mode_without_exposing_path(tmp_path) -> None:
    worker = Crawl4AIWorker(
        "worker.sock",
        {"instagram.com"},
        user_data_dir=str(tmp_path),
    )
    worker._expire_sessions = AsyncMock()
    worker._browser_context_alive = MagicMock(return_value=True)

    response = await worker.dispatch({"op": "health"})

    assert response["authenticated_profile"] is True
    assert response["visible_browser"] is False
    assert response["browser_channel"] == "chromium"
    assert response["browser_ready"] is True
    assert str(tmp_path) not in json.dumps(response)


@pytest.mark.asyncio
async def test_worker_health_reports_visible_browser_mode(tmp_path) -> None:
    worker = Crawl4AIWorker(
        "worker.sock",
        {"facebook.com"},
        user_data_dir=str(tmp_path),
        headless=False,
    )
    worker._expire_sessions = AsyncMock()
    worker._browser_context_alive = MagicMock(return_value=True)

    response = await worker.dispatch({"op": "health"})

    assert response["visible_browser"] is True


@pytest.mark.asyncio
async def test_worker_health_reports_configured_browser_channel(tmp_path) -> None:
    worker = Crawl4AIWorker(
        "worker.sock",
        {"facebook.com"},
        user_data_dir=str(tmp_path),
        browser_channel="msedge",
    )
    worker._expire_sessions = AsyncMock()
    worker._browser_context_alive = MagicMock(return_value=False)

    response = await worker.dispatch({"op": "health"})

    assert response["browser_channel"] == "msedge"


@pytest.mark.asyncio
async def test_worker_health_reports_closed_browser_without_stale_ready_state(tmp_path) -> None:
    worker = Crawl4AIWorker(
        "worker.sock",
        {"facebook.com"},
        user_data_dir=str(tmp_path),
        headless=False,
    )
    worker.sessions["stale"] = BrowserSession("https://www.facebook.com/", 1)
    worker._crawler = MagicMock()
    worker._browser_context_alive = MagicMock(return_value=False)

    response = await worker.dispatch({"op": "health"})

    assert response["status"] == "browser_closed"
    assert response["browser_ready"] is False
    assert response["active_sessions"] == 0


@pytest.mark.asyncio
async def test_worker_starts_idle_without_launching_chromium(tmp_path) -> None:
    worker = Crawl4AIWorker(
        "worker.sock",
        {"facebook.com"},
        user_data_dir=str(tmp_path),
        headless=False,
    )
    worker._validate_browser_setup = MagicMock()
    worker._launch_crawler = AsyncMock()

    await worker.start()
    response = await worker.dispatch({"op": "health"})

    worker._launch_crawler.assert_not_awaited()
    assert response["status"] == "idle"
    assert response["browser_ready"] is False


@pytest.mark.asyncio
async def test_worker_recovers_closed_browser_before_next_operation(tmp_path) -> None:
    worker = Crawl4AIWorker(
        "worker.sock",
        {"facebook.com"},
        user_data_dir=str(tmp_path),
        headless=False,
    )
    worker._browser_context_alive = MagicMock(side_effect=[False, True])
    worker._restart_browser = AsyncMock()
    worker._expire_sessions = AsyncMock()

    response = await worker.dispatch({"op": "reset"})

    assert response == {"ok": True, "reset_sessions": 0}
    worker._restart_browser.assert_not_awaited()

    worker._browser_context_alive = MagicMock(return_value=False)
    await worker._ensure_browser_ready()
    worker._restart_browser.assert_awaited_once()


@pytest.mark.asyncio
async def test_social_crawl_tool_transports_worker_html(monkeypatch) -> None:
    html = '<section><a href="/source">Source link</a></section>'
    monkeypatch.setattr(
        SocialCrawlClient,
        "request_async",
        AsyncMock(return_value={"ok": True, "session_id": "s1", "html": html}),
    )

    output = await SocialCrawlTool().execute("open", url="https://www.instagram.com/")

    assert "CRAWL_SESSION_ID: s1" in output
    assert f"--- BEGIN RENDERED HTML ---\n{html}\n--- END RENDERED HTML ---" in output
    assert '\\"/source\\"' not in output
    SocialCrawlClient.request_async.assert_awaited_once_with(
        "open", url="https://www.instagram.com/"
    )


@pytest.mark.asyncio
async def test_social_crawl_tool_returns_screenshot_with_html(monkeypatch) -> None:
    monkeypatch.setattr(
        SocialCrawlClient,
        "request_async",
        AsyncMock(return_value={
            "ok": True,
            "session_id": "s1",
            "url": "https://www.instagram.com/",
            "status_code": 200,
            "html": "<main>post</main>",
            "screenshot_base64": "aW1hZ2U=",
            "screenshot_mime": "image/jpeg",
        }),
    )

    output = await SocialCrawlTool().execute(
        "open", url="https://www.instagram.com/"
    )

    assert output[0]["type"] == "text"
    assert "<main>post</main>" in output[0]["text"]
    assert output[1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/jpeg;base64,aW1hZ2U="},
    }


@pytest.mark.asyncio
async def test_social_crawl_tool_does_not_expose_manual_close(monkeypatch) -> None:
    request = AsyncMock(return_value={
        "ok": True, "session_id": "s1", "html": "<main>page</main>"
    })
    monkeypatch.setattr(SocialCrawlClient, "request_async", request)
    tool = SocialCrawlTool()

    await tool.execute("open", url="https://www.instagram.com/")
    close_result = await tool.execute("close")

    assert len(request.await_args_list) == 1
    assert close_result.startswith("Error: browser cleanup is automatic")
    assert "close" not in SocialCrawlTool.parameters["properties"]["action"]["enum"]


@pytest.mark.asyncio
async def test_social_crawl_tool_navigates_same_session_and_rejects_repeat(monkeypatch) -> None:
    request = AsyncMock(side_effect=[
        {"ok": True, "session_id": "s1", "html": "<main>one</main>"},
        {"ok": True, "session_id": "s1", "html": "<main>two</main>"},
    ])
    monkeypatch.setattr(SocialCrawlClient, "request_async", request)
    tool = SocialCrawlTool()

    await tool.execute("open", url="https://www.facebook.com/one")
    await tool.execute("navigate", url="https://www.instagram.com/two")
    repeated = await tool.execute("navigate", url="https://www.instagram.com/two")

    assert request.await_args_list[1].args == ("navigate",)
    assert request.await_args_list[1].kwargs["session_id"] == "s1"
    assert repeated.startswith("URL_ALREADY_VISITED")


@pytest.mark.asyncio
async def test_repeated_open_becomes_navigation_and_stale_session_is_ignored(monkeypatch) -> None:
    request = AsyncMock(side_effect=[
        {
            "ok": True,
            "session_id": "current",
            "url": "https://www.facebook.com/one",
            "html": "<main>one</main>",
        },
        {
            "ok": True,
            "session_id": "current",
            "url": "https://www.instagram.com/two",
            "html": "<main>two</main>",
        },
        {
            "ok": True,
            "session_id": "current",
            "url": "https://www.instagram.com/two",
            "html": "<main>after scroll</main>",
        },
    ])
    monkeypatch.setattr(SocialCrawlClient, "request_async", request)
    tool = SocialCrawlTool()

    await tool.execute("open", url="https://www.facebook.com/one")
    second = await tool.execute("open", url="https://www.instagram.com/two")
    await tool.execute("scroll", session_id="obsolete", scroll_pixels=900)

    assert request.await_args_list[1].args == ("navigate",)
    assert request.await_args_list[1].kwargs["session_id"] == "current"
    assert request.await_args_list[2].kwargs["session_id"] == "current"
    assert "PRESERVED EARLIER PAGE EVIDENCE" in second
    assert "https://www.facebook.com/one" in second


@pytest.mark.asyncio
async def test_browser_action_budget_reserves_time_for_final_findings(monkeypatch) -> None:
    request = AsyncMock(return_value={
        "ok": True,
        "session_id": "s1",
        "url": "https://www.facebook.com/one",
        "html": "<main>evidence</main>",
    })
    monkeypatch.setattr(SocialCrawlClient, "request_async", request)
    tool = SocialCrawlTool()

    await tool.execute("open", url="https://www.facebook.com/one")
    for _ in range(13):
        result = await tool.execute("scroll", scroll_pixels=100)
    exhausted = await tool.execute("scroll", scroll_pixels=100)

    assert "ACTION_GUIDANCE" in result
    assert exhausted.startswith("BROWSER_ACTION_BUDGET_EXHAUSTED")
    assert request.await_count == 14


@pytest.mark.asyncio
async def test_social_crawl_cleanup_resets_orphaned_worker_session(monkeypatch) -> None:
    request = AsyncMock(return_value={"ok": True, "reset_sessions": 1})
    monkeypatch.setattr(SocialCrawlClient, "request_async", request)
    tool = SocialCrawlTool()

    await tool.prepare()
    await tool.cleanup()

    assert [call.args for call in request.await_args_list] == [("reset",), ("reset",)]


@pytest.mark.asyncio
async def test_worker_reset_closes_every_active_session() -> None:
    worker = Crawl4AIWorker("worker.sock", {"instagram.com"})
    worker.sessions = {
        "s1": BrowserSession("https://www.instagram.com/a", 1),
        "s2": BrowserSession("https://www.instagram.com/b", 1),
    }
    worker._expire_sessions = AsyncMock()
    worker._browser_context_alive = MagicMock(return_value=True)
    worker._crawler = AsyncMock()

    async def kill_session(session_id: str) -> None:
        worker.sessions.pop(session_id)

    worker._kill_session = AsyncMock(side_effect=kill_session)

    response = await worker.dispatch({"op": "reset"})

    assert response == {"ok": True, "reset_sessions": 2}
    assert worker.sessions == {}
    assert worker._crawler is None
