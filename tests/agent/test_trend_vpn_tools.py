from __future__ import annotations

from nanobot.agent.tools.trend_vpn import (
    TrendVpnBrowserFetchTool,
    TrendVpnFetchTool,
    TrendVpnSessionCloseTool,
    TrendVpnSessionStartTool,
)


def test_trend_vpn_tools_are_serialized_and_have_narrow_schemas():
    tools = [
        TrendVpnSessionStartTool(),
        TrendVpnFetchTool(),
        TrendVpnBrowserFetchTool(),
        TrendVpnSessionCloseTool(),
    ]
    assert [tool.name for tool in tools] == [
        "trend_vpn_session_start",
        "trend_vpn_fetch",
        "trend_vpn_browser_fetch",
        "trend_vpn_session_close",
    ]
    assert all(tool.supports_parallel_calls is False for tool in tools)
    assert "session_id" in TrendVpnFetchTool.parameters["required"]
    assert "url" in TrendVpnFetchTool.parameters["required"]
    assert TrendVpnFetchTool.parameters["properties"]["url"]["type"] == "string"
    assert TrendVpnFetchTool.parameters["properties"]["max_chars"]["maximum"] == 120000
    assert "session_id" in TrendVpnBrowserFetchTool.parameters["required"]
    assert "url" in TrendVpnBrowserFetchTool.parameters["required"]
