from pydantic import BaseModel, Field
from typing import List, Tuple, Set

# from langchain.schema import AgentAction, ChatMessage
from agentverse.utils import AgentAction


class Message(BaseModel):
    content: str = Field(default="")
    sender: str = Field(default="")
    receiver: Set[str] = Field(default=set({"all"}))
    tool_response: List[Tuple[AgentAction, str]] = Field(default=[])


class StructuredPrompt(BaseModel):
    system_content: str = Field(default="", description="放在message system字段，通常包含question和基础指令")
    user_content: str = Field(default="", description="放在message user字段，通常包含role和当前回合指令")
    
    def __init__(self, system_content: str = "", user_content: str = "", **kwargs):
        super().__init__(system_content=system_content, user_content=user_content, **kwargs)
