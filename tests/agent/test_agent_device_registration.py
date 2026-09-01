from unittest.mock import MagicMock

from nanobot.agent.delegation import ForegroundAgentManager
from nanobot.config.schema import AgentDeviceConfig


def _provider():
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    return provider


def test_loop_registers_agent_device_by_default(tmp_path) -> None:
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus

    loop = AgentLoop(bus=MessageBus(), provider=_provider(), workspace=tmp_path)
    assert loop.tools.get("agent_device") is not None


def test_loop_skips_agent_device_when_disabled(tmp_path) -> None:
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus

    loop = AgentLoop(
        bus=MessageBus(),
        provider=_provider(),
        workspace=tmp_path,
        agent_device_config=AgentDeviceConfig(enabled=False),
    )
    assert loop.tools.get("agent_device") is None


def test_foreground_role_profiles_scope_device_without_write_tools(tmp_path) -> None:
    manager = ForegroundAgentManager(provider=_provider(), workspace=tmp_path)
    tools = manager._tools_for_role("explorer")

    assert tools.get("agent_device") is not None
    assert tools.get("write_file") is None
    assert tools.get("edit_file") is None
    assert manager._tools_for_role("planner").get("agent_device") is None


def test_foreground_worker_has_scoped_write_tools(tmp_path) -> None:
    manager = ForegroundAgentManager(provider=_provider(), workspace=tmp_path)
    scopes = manager._normalize_scopes(tmp_path, ["report/"])
    tools = manager._worker_tools("worker-1", scopes)

    assert tools.get("write_file") is not None
    assert tools.get("edit_file") is not None
    assert tools.get("message") is None
    assert tools.get("cron") is None
    assert tools.get("delegate_task") is None
