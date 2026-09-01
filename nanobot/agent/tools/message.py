"""Message tool for sending messages to users."""

from typing import Any, Awaitable, Callable

from nanobot.agent.tools.base import Tool
from nanobot.agent.turn import ToolOutcome, TurnContext
from nanobot.bus.events import OutboundMessage


class MessageTool(Tool):
    """Tool to send messages to users on chat channels."""

    def __init__(
        self,
        send_callback: Callable[[OutboundMessage], Awaitable[None]] | None = None,
        default_channel: str = "",
        default_chat_id: str = "",
        default_message_id: str | None = None,
    ):
        self._send_callback = send_callback
        self._default_channel = default_channel
        self._default_chat_id = default_chat_id
        self._default_message_id = default_message_id
        self._sent_in_turn: bool = False
        self._sent_messages_in_turn: list[OutboundMessage] = []

    def set_context(self, channel: str, chat_id: str, message_id: str | None = None) -> None:
        """Set the current message context."""
        self._default_channel = channel
        self._default_chat_id = chat_id
        self._default_message_id = message_id

    def set_send_callback(self, callback: Callable[[OutboundMessage], Awaitable[None]]) -> None:
        """Set the callback for sending messages."""
        self._send_callback = callback

    def start_turn(self) -> None:
        """Reset per-turn send tracking."""
        self._sent_in_turn = False
        self._sent_messages_in_turn = []

    @property
    def sent_messages_in_turn(self) -> tuple[OutboundMessage, ...]:
        """Return messages successfully sent during the current agent turn."""
        return tuple(self._sent_messages_in_turn)

    @property
    def name(self) -> str:
        return "message"

    @property
    def description(self) -> str:
        return (
            "Send a message to the user, optionally with file attachments. "
            "This is the ONLY way to deliver files (images, documents, audio, video) to the user. "
            "Use the 'media' parameter with file paths to attach files. "
            "Do NOT use read_file to send files — that only reads content for your own analysis."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The message content to send"
                },
                "channel": {
                    "type": "string",
                    "description": "Optional: target channel (telegram, discord, etc.)"
                },
                "chat_id": {
                    "type": "string",
                    "description": "Optional: target chat/user ID"
                },
                "media": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional: list of file paths to attach (images, audio, documents)"
                }
            },
            "required": ["content"]
        }

    async def execute(
        self,
        content: str,
        channel: str | None = None,
        chat_id: str | None = None,
        message_id: str | None = None,
        media: list[str] | None = None,
        **kwargs: Any
    ) -> str:
        channel = channel or self._default_channel
        chat_id = chat_id or self._default_chat_id
        message_id = message_id or self._default_message_id

        if not channel or not chat_id:
            return "Error: No target channel/chat specified"

        if not self._send_callback:
            return "Error: Message sending not configured"

        msg = OutboundMessage(
            channel=channel,
            chat_id=chat_id,
            content=content,
            media=media or [],
            metadata={
                "message_id": message_id,
            },
        )

        try:
            await self._send_callback(msg)
            self._sent_messages_in_turn.append(msg)
            if channel == self._default_channel and chat_id == self._default_chat_id:
                self._sent_in_turn = True
            media_info = f" with {len(media)} attachments" if media else ""
            return f"Message sent to {channel}:{chat_id}{media_info}"
        except Exception as e:
            return f"Error sending message: {str(e)}"

    async def execute_with_context(
        self,
        context: TurnContext,
        content: str,
        channel: str | None = None,
        chat_id: str | None = None,
        message_id: str | None = None,
        media: list[str] | None = None,
        **kwargs: Any,
    ) -> ToolOutcome:
        """Send through the current turn's delivery state.

        The shared ``MessageTool`` instance is retained for compatibility with
        older callers, but canonical turns never mutate its routing or sent
        ledger.  Every value used for routing and suppression comes from the
        explicit context passed by the turn registry.
        """
        target = context.delivery.primary or context.delivery.target
        channel = channel or (target.channel if target else "")
        chat_id = chat_id or (
            target.chat_id or target.to or target.recipient if target else ""
        )
        message_id = message_id or (target.message_id if target else None)

        if not channel or not chat_id:
            return ToolOutcome(content="Error: No target channel/chat specified")
        if not self._send_callback:
            return ToolOutcome(content="Error: Message sending not configured")

        msg = OutboundMessage(
            channel=channel,
            chat_id=chat_id,
            content=content,
            media=media or [],
            metadata={"message_id": message_id},
        )
        try:
            await self._send_callback(msg)
        except Exception as exc:
            return ToolOutcome(content=f"Error sending message: {exc}")

        context.delivery.sent_messages.append(msg)
        primary = context.delivery.primary or context.delivery.target
        if primary is not None:
            primary_chat = primary.chat_id or primary.to or primary.recipient
            if channel == primary.channel and chat_id == primary_chat:
                context.delivery.delivered = True
                context.delivery.message_id = message_id
        return ToolOutcome(
            content=(
                f"Message sent to {channel}:{chat_id}"
                f" with {len(media)} attachments" if media
                else f"Message sent to {channel}:{chat_id}"
            )
        )
