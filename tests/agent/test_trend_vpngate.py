from __future__ import annotations

import base64
import csv
import io

from nanobot.agent.tools.trend_vpngate import parse_api_response


PROFILE = """client
dev tun
proto tcp
remote public-vpn-1.opengw.net 443
<ca>
CA
</ca>
<cert>
CERT
</cert>
<key>
KEY
</key>
"""


def _api_row(hostname: str, ip_address: str, score: int) -> str:
    values = [
        hostname,
        ip_address,
        str(score),
        "12",
        "1000000",
        "Malaysia",
        "MY",
        "1",
        "99",
        "2",
        "3",
        "",
        "",
        "",
        "",
        base64.b64encode(PROFILE.encode()).decode(),
    ]
    return ",".join(values)


def test_agent_ranks_and_decodes_vpngate_candidates() -> None:
    header = "#" + ",".join(
        [
            "HostName",
            "IP",
            "Score",
            "Ping",
            "Speed",
            "CountryLong",
            "CountryShort",
            "NumVpnSessions",
            "Uptime",
            "TotalUsers",
            "TotalTraffic",
            "LogType",
            "Operator",
            "Message",
            "OpenVPN_ConfigData_Base64",
        ]
    )
    # Keep the fixture generated with csv so commas in future profile metadata
    # do not accidentally make this test depend on hand-escaped rows.
    row = io.StringIO()
    writer = csv.writer(row, lineterminator="")
    writer.writerow(
        [
            "public-vpn-1",
            "219.100.37.10",
            "20",
            "12",
            "1000000",
            "Malaysia",
            "MY",
            "1",
            "99",
            "2",
            "3",
            "",
            "",
            "",
            base64.b64encode(PROFILE.encode()).decode(),
        ]
    )
    candidates = parse_api_response(
        "\n".join(["*vpn_servers", header, row.getvalue()]), max_candidates=1
    )
    assert len(candidates) == 1
    assert candidates[0].hostname == "public-vpn-1"
    assert candidates[0].profile_text == PROFILE
