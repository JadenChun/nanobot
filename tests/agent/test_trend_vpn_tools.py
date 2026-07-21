from __future__ import annotations

from nanobot.agent.tools.trend_vpn import (
    TrendVpnFetchTool,
    TrendVpnSessionCloseTool,
    TrendVpnSessionStartTool,
)


def test_trend_vpn_tools_are_serialized_and_have_narrow_schemas():
    tools = [TrendVpnSessionStartTool(), TrendVpnFetchTool(), TrendVpnSessionCloseTool()]
    assert [tool.name for tool in tools] == [
        "trend_vpn_session_start",
        "trend_vpn_fetch",
        "trend_vpn_session_close",
    ]
    assert all(tool.supports_parallel_calls is False for tool in tools)
    assert "session_id" in TrendVpnFetchTool.parameters["required"]
    assert "url" in TrendVpnFetchTool.parameters["required"]
    assert TrendVpnFetchTool.parameters["properties"]["url"]["type"] == "string"
