from nanobot.bus.events import OutboundMessage
from nanobot.cron.delivery import build_explicit_fanout_messages, build_result_messages
from nanobot.cron.types import CronDestination, CronPayload


def test_delivery_destinations_include_primary_and_deduplicate() -> None:
    payload = CronPayload(
        deliver=True,
        channel="telegram",
        to="6344587670",
        additional_destinations=[
            CronDestination("telegram", "-1001234567890"),
            CronDestination("telegram", "6344587670"),
            CronDestination("telegram", "-1001234567890"),
        ],
    )

    assert payload.delivery_destinations() == [
        CronDestination("telegram", "6344587670"),
        CronDestination("telegram", "-1001234567890"),
    ]


def test_disabled_payload_has_no_delivery_destinations() -> None:
    payload = CronPayload(
        deliver=False,
        channel="telegram",
        to="6344587670",
        additional_destinations=[CronDestination("telegram", "-1001234567890")],
    )

    assert payload.delivery_destinations() == []


def test_result_messages_are_built_for_each_missing_destination() -> None:
    destinations = [
        CronDestination("telegram", "6344587670"),
        CronDestination("telegram", "-1001234567890"),
    ]
    sent = [OutboundMessage("telegram", "-1001234567890", "Report ready")]

    assert build_result_messages("Report ready", destinations, sent) == [
        OutboundMessage("telegram", "6344587670", "Report ready"),
    ]


def test_explicit_primary_message_with_media_is_copied_to_other_destinations() -> None:
    destinations = [
        CronDestination("telegram", "6344587670"),
        CronDestination("telegram", "-1001234567890"),
    ]
    sent = [
        OutboundMessage(
            "telegram",
            "6344587670",
            "Weekly report",
            media=["/tmp/report.pdf"],
            metadata={"message_id": "123"},
        ),
    ]

    assert build_explicit_fanout_messages(destinations, sent) == [
        OutboundMessage(
            "telegram",
            "-1001234567890",
            "Weekly report",
            media=["/tmp/report.pdf"],
        ),
    ]


def test_explicit_fanout_does_not_duplicate_an_existing_send() -> None:
    destinations = [
        CronDestination("telegram", "6344587670"),
        CronDestination("telegram", "-1001234567890"),
    ]
    sent = [
        OutboundMessage("telegram", "6344587670", "Weekly report"),
        OutboundMessage("telegram", "-1001234567890", "Weekly report"),
    ]

    assert build_explicit_fanout_messages(destinations, sent) == []
