from __future__ import annotations

import re
from typing import Union

from agentverse.parser import OutputParser, LLMResult
from agentverse.utils import AgentAction, AgentFinish
from agentverse.parser import OutputParserError, output_parser_registry


@output_parser_registry.register("cot")
class COTParser(OutputParser):
    """
    Output parser for Chain-of-Thought (COT) agents that extract answer and reasoning
    from structured LLM responses.
    
    Expected format:
    <Reasoning>YOUR_REASONING<Answer>ANSWER_LETTER
    """
    
    def parse(self, output: LLMResult) -> Union[AgentAction, AgentFinish]:
        """
        Parse LLM output to extract answer and reasoning
        
        Args:
            output: LLM result containing the response content
            
        Returns:
            AgentFinish with parsed answer and reasoning
        """
        text = output.content
        cleaned_output = text.strip()
        answer, reasoning = self._extract_answer_and_reasoning(cleaned_output)
        validated_answer = self._validate_answer(answer)
        
        return AgentFinish(
            return_values={
                "answer": validated_answer,
                "reasoning": reasoning,
                "output": cleaned_output
            },
            log=cleaned_output
        )
    
    def _extract_answer_and_reasoning(self, text: str) -> tuple[str, str]:
        """
        Extract answer and reasoning from text using <Reasoning>...<Answer>... format
        
        Args:
            text: Raw text from LLM
            
        Returns:
            Tuple of (answer, reasoning)
        """
        pattern = r'<Reasoning>\s*(.*?)\s*<Answer>\s*(.*?)(?:\n|$)'
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        
        if match:
            reasoning = match.group(1).strip()
            answer = match.group(2).strip()
            return answer, reasoning
        
        # If no match found, return empty answer with full text as reasoning
        return "", text.strip()
    
    def _validate_answer(self, answer: str) -> str:
        """
        Validate and normalize answer format
        
        Args:
            answer: Raw answer extracted from text
            
        Returns:
            Validated answer string
        """
        if not answer:
            return ""
        
        # Convert to uppercase for consistency
        answer = answer.upper()
        
        # Valid single letter answers (support A-Z for flexibility)
        if len(answer) == 1 and answer.isalpha():
            return answer
        
        # Valid True/False/Unknown answers
        answer_lower = answer.lower()
        if answer_lower in ['true', 'false', 'unknown']:
            return answer_lower.capitalize()
        
        # Handle common variations
        if answer.startswith('TRUE'):
            return 'true'
        elif answer.startswith('FALSE'):
            return 'false'
        elif answer.startswith('UNKNOWN'):
            return 'unknown'
        
        # Return as-is if not recognized (for flexibility)
        return answer
    
    def _clean_reasoning(self, reasoning: str) -> str:
        if not reasoning:
            return ""
        
        # Remove excessive whitespace
        reasoning = re.sub(r'\s+', ' ', reasoning)
        
        # Remove common prefixes
        reasoning = re.sub(r'^(Reasoning|REASONING|Reason|REASON)\s*:?\s*', '', reasoning)
        
        return reasoning.strip()


@output_parser_registry.register("plan_and_solve")
class PlanAndSolveParser(OutputParser):
    """
    Output parser for Plan-and-Solve agents that extract answer and reasoning
    from natural language responses.
    
    Expected pattern:
    Natural language reasoning followed by "Therefore, the answer is [answer]"
    """
    
    def parse(self, output: LLMResult) -> Union[AgentAction, AgentFinish]:
        """
        Parse Plan-and-Solve output to extract answer and reasoning
        
        Args:
            output: LLM result containing the natural language response
            
        Returns:
            AgentFinish with parsed answer and reasoning
        """
        text = output.content
        cleaned_output = text.strip()
        answer, reasoning = self._extract_answer_and_reasoning(cleaned_output)
        validated_answer = self._validate_answer(answer)
        
        return AgentFinish(
            return_values={
                "answer": validated_answer,
                "reasoning": reasoning,
                "output": cleaned_output
            },
            log=cleaned_output
        )
    
    def _extract_answer_and_reasoning(self, text: str) -> tuple[str, str]:
        """
        Extract answer and reasoning from Plan-and-Solve response with <answer>ANSWER_LETTER<answer/> format
        
        Args:
            text: Raw text from LLM
            
        Returns:
            Tuple of (answer, reasoning)
        """
        # Look for the standardized <answer>ANSWER_LETTER<answer/> format
        pattern = r'<answer>\s*(.*?)\s*<answer/>'
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        
        if match:
            answer = match.group(1).strip()
            # Use text before the answer tag as reasoning
            reasoning = text[:match.start()].strip()
            return answer, reasoning
        
        # Fallback: look for natural language patterns for backward compatibility
        fallback_patterns = [
            r'<answer>\s*([^.\n]*)',
            r'Therefore,?\s*the answer is\s*([^.\n]*)',
            r'The answer is\s*([^.\n]*)',
            r'the answer is\s*([^.\n]*)'
        ]
        
        for pattern in fallback_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
                reasoning = text[:match.start()].strip()
                return answer, reasoning
        
        # Final fallback: no clear answer found, return empty answer with full text as reasoning
        return text.strip(), text.strip()
    
    def _validate_answer(self, answer: str) -> str:
        """
        Validate and normalize answer format for Plan-and-Solve
        
        Args:
            answer: Raw answer extracted from text
            
        Returns:
            Validated answer string
        """
        if not answer:
            return ""
        
        # Clean up the answer string
        answer = answer.strip().rstrip('.').strip()
        
        # Convert to uppercase for consistency
        answer = answer.upper()
        
        # Valid single letter answers (support A-Z for flexibility)
        if len(answer) == 1 and answer.isalpha():
            return answer
        
        # Valid True/False/Unknown answers
        answer_lower = answer.lower()
        if answer_lower in ['true', 'false', 'unknown']:
            return answer_lower.capitalize()
        
        # Handle common variations for True/False/Unknown
        if answer.startswith('TRUE') or any(word in answer_lower for word in ['true', 'correct', 'yes']):
            return 'True'
        elif answer.startswith('FALSE') or any(word in answer_lower for word in ['false', 'incorrect', 'no']):
            return 'False'
        elif answer.startswith('UNKNOWN') or any(word in answer_lower for word in ['unknown', 'uncertain', 'unclear']):
            return 'Unknown'
        
        # Extract first valid letter if answer contains more text
        match = re.search(r'[A-Z]', answer)
        if match:
            return match.group(0)
        
        # Return as-is if not recognized (for flexibility)
        return answer
