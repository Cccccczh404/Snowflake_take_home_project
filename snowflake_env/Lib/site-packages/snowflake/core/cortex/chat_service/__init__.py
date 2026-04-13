from snowflake.core.cortex.chat_service._generated.models.chat_request_messages_inner import ChatRequestMessagesInner

from ._chat_service import ChatRequest, CortexChatService
from ._generated import CortexChatApi


CortexChatServiceApi = CortexChatApi

__all__ = [
    "CortexChatService",
    "ChatRequest",
    "CortexChatServiceApi",
    "ChatRequestMessagesInner",
]
