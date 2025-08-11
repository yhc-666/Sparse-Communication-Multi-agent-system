from __future__ import annotations

import re
from typing import Union

from agentverse.parser import OutputParser, LLMResult

from agentverse.utils import AgentAction, AgentFinish

from agentverse.parser import OutputParserError, output_parser_registry


@output_parser_registry.register("final_debate")
class FinalDebateParser(OutputParser):
    def parse(self, output: LLMResult, cnt_turn: int, max_turns: int, agent_nums: int) -> Union[AgentAction, AgentFinish]:
        text = output.content
        cleaned_output = text.strip()
        cleaned_output = re.sub(r"\n+", "\n", cleaned_output)
        
        # In final turn, extract the answer from <answer> tags
        if cnt_turn >= max_turns - agent_nums:
            final_answer = self._extract_final_answer(cleaned_output)
            return AgentFinish(
                return_values={
                    "output": cleaned_output,
                    "final_answer": final_answer
                }, 
                log=cleaned_output
            )
        
        # Non-final turn processing
        return AgentFinish({"output": cleaned_output}, cleaned_output)
    
    def _extract_final_answer(self, text: str) -> str:
        """
        Extract final answer from <answer>OPTION format
        
        Args:
            text: Raw text from LLM response
            
        Returns:
            Extracted answer option (A, B, C, D, E, F, G, ...) or mapped answer
        """
        # Pattern to match <answer>OPTION</answer> format
        pattern = r'<answer>\s*([^<]+)\s*</answer>'
        match = re.search(pattern, text, re.IGNORECASE)
        
        if match:
            answer = match.group(1).strip()
            # Map common answer formats to standard options
            answer_mapping = {
                "true": "A",
                "false": "B", 
                "unknown": "C",
                "a": "A", "b": "B", "c": "C", "d": "D", "e": "E",
                "f": "F", "g": "G", "h": "H", "i": "I", "j": "J"
            }
            
            # Check if it's already a standard option (support A-Z for flexibility)
            if len(answer) == 1 and answer.upper().isalpha():
                return answer.upper()
            
            # Try to map from text to option
            if answer.lower() in answer_mapping:
                return answer_mapping[answer.lower()]
            
            # Return the original answer if no mapping found
            return answer.upper()
        
        # Fallback: try to match simple <answer>OPTION format without closing tag
        pattern = r'<answer>\s*([A-Z])\s*'
        match = re.search(pattern, text, re.IGNORECASE)
        
        if match:
            return match.group(1).upper()
        
        # If no standard format found, return empty string
        return ""

