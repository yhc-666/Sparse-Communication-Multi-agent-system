from __future__ import annotations

import re
from typing import Union, Optional

from agentverse.parser import OutputParser, LLMResult
from pydantic import Field

from agentverse.utils import AgentAction, AgentFinish

from agentverse.parser import output_parser_registry


@output_parser_registry.register("translate")
class TranslateParser(OutputParser):
    dataset_name: str = Field(default="ProofWriter")
    
    def parse(self, output: LLMResult, cnt_turn: int, max_turns: int, agent_nums: int, agent_name: Optional[str] = None) -> Union[AgentAction, AgentFinish]:
        text = output.content
        cleaned_output = text.strip()
        cleaned_output = re.sub(r"\n+", "\n", cleaned_output)
        cleaned_output = re.sub(r"\n-", "\n", cleaned_output)  # Remove dash after newline
        cleaned_output = cleaned_output.replace("**", "")
        
        return AgentFinish({"output": cleaned_output}, cleaned_output)
