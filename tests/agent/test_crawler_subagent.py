from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.delegation import ForegroundAgentManager
from nanobot.agent.tools.crawler import CrawlResearchTool
from nanobot.agent.tools.social_crawl import SocialCrawlTool
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import Config, CrawlerAgentConfig
from nanobot.nanobot import _make_crawler_provider


def _manager(tmp_path, **kwargs) -> ForegroundAgentManager:
    provider = MagicMock()
    provider.get_default_model.return_value = "main-model"
    return ForegroundAgentManager(
        provider=provider,
        workspace=tmp_path,
        **kwargs,
    )


def test_crawler_config_accepts_camel_case_limits() -> None:
    config = Config(
        agents={
            "crawler": {
                "enabled": True,
                "provider": "openrouter",
                "model": "vendor/cheap-crawler-model",
                "maxTokens": {"input": 18000, "output": 1600},
                "maxToolIterations": 12,
                "reasoningEffort": "low",
            }
        }
    )

    assert config.agents.crawler.enabled is True
    assert config.agents.crawler.provider == "openrouter"
    assert config.agents.crawler.model == "vendor/cheap-crawler-model"
    assert config.agents.crawler.max_tokens.input == 18000
    assert config.agents.crawler.max_tokens.output == 1600
    assert config.agents.crawler.max_tool_iterations == 12


def test_crawler_role_can_use_a_provider_separate_from_main_agent(tmp_path) -> None:
    crawler_provider = MagicMock()
    manager = _manager(tmp_path, crawler_provider=crawler_provider)

    assert manager.crawler_runner.provider is crawler_provider
    assert manager.runner.provider is not crawler_provider


def test_main_loop_exposes_only_foreground_crawler_delegation(tmp_path, monkeypatch) -> None:
    from nanobot.agent.loop import AgentLoop

    monkeypatch.setenv("CRAWL4AI_WORKER_ENABLED", "true")
    provider = MagicMock()
    provider.get_default_model.return_value = "main-model"

    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        crawler_agent_config=CrawlerAgentConfig(enabled=True),
    )

    assert loop.tools.get("crawl_research") is not None
    assert loop.tools.get("spawn") is None
    assert loop.tools.get("spawn_crawler") is None
    assert loop.tools.get("spawn_pipeline") is None


def test_crawler_provider_is_created_from_its_own_provider_config(monkeypatch) -> None:
    monkeypatch.setenv("CRAWL4AI_WORKER_ENABLED", "true")
    config = Config(
        agents={
            "defaults": {"provider": "anthropic", "model": "anthropic/main-model"},
            "crawler": {
                "enabled": True,
                "provider": "openrouter",
                "model": "vendor/cheap-crawler-model",
            },
        },
        providers={
            "anthropic": {"apiKey": "main-key"},
            "openrouter": {"apiKey": "sk-or-crawler-key"},
        },
    )

    crawler_provider = _make_crawler_provider(config)

    assert crawler_provider is not None
    assert crawler_provider.default_model == "vendor/cheap-crawler-model"


@pytest.mark.asyncio
async def test_foreground_crawler_uses_its_own_model_and_budget(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CRAWL4AI_WORKER_ENABLED", "true")
    manager = _manager(
        tmp_path,
        crawler_model="openrouter/cheap-crawler-model",
        crawler_max_iterations=9,
        crawler_max_input_tokens=17000,
        crawler_max_output_tokens=1400,
        crawler_reasoning_effort="low",
    )
    manager.crawler_runner.run = AsyncMock(return_value=MagicMock(
        stop_reason="completed",
        final_content="finished",
        tool_events=[],
    ))
    monkeypatch.setattr(SocialCrawlTool, "prepare", AsyncMock())
    monkeypatch.setattr(SocialCrawlTool, "cleanup", AsyncMock())

    result = await manager.run_crawler(task="Inspect the supplied public page.")

    assert result == "finished"
    spec = manager.crawler_runner.run.await_args.args[0]
    assert spec.model == "openrouter/cheap-crawler-model"
    assert spec.max_iterations == 9
    assert spec.max_input_tokens == 17000
    assert spec.max_tokens == 1400
    assert spec.reasoning_effort == "low"
    assert spec.concurrent_tools is False
    assert spec.tools.tool_names == ["social_crawl"]
    assert "compact rendered HTML" in spec.initial_messages[0]["content"]
    assert "manages the session internally" in spec.initial_messages[0]["content"]
    assert "viewport screenshot" in spec.initial_messages[0]["content"]
    assert "Never\nclose the browser yourself" in spec.initial_messages[0]["content"]


@pytest.mark.asyncio
async def test_authenticated_crawler_prompt_is_authorized_read_only(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CRAWL4AI_WORKER_ENABLED", "true")
    monkeypatch.setenv("CRAWL4AI_AUTH_PROFILE_ENABLED", "true")
    manager = _manager(tmp_path)
    manager.crawler_runner.run = AsyncMock(
        return_value=MagicMock(stop_reason="completed", final_content="finished", tool_events=[])
    )
    monkeypatch.setattr(SocialCrawlTool, "prepare", AsyncMock())
    monkeypatch.setattr(SocialCrawlTool, "cleanup", AsyncMock())

    await manager.run_crawler(task="Inspect public comments visible after login.")

    prompt = manager.crawler_runner.run.await_args.args[0].initial_messages[0]["content"]
    assert "operator-prepared authenticated browser profile" in prompt
    assert "including non-public content visible to the signed-in account" in prompt
    assert "never request, expose, or enter them" in prompt
    assert "outside" in prompt
    assert "profile's authorized scope" in prompt
    assert "Do not open DMs" in prompt
    assert "Clicking is disabled" in prompt


@pytest.mark.asyncio
async def test_crawler_tool_waits_for_foreground_result() -> None:
    manager = MagicMock()
    manager.run_crawler = AsyncMock(return_value="findings")
    tool = CrawlResearchTool(manager)

    result = await tool.execute("Inspect this public Instagram post")

    assert result == "findings"
    manager.run_crawler.assert_awaited_once_with(task="Inspect this public Instagram post")


@pytest.mark.asyncio
async def test_foreground_crawler_is_rejected_when_worker_is_disabled(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CRAWL4AI_WORKER_ENABLED", "false")
    manager = _manager(tmp_path)

    result = await manager.run_crawler(task="Inspect a page")

    assert result == "Error: crawler worker integration is disabled"
