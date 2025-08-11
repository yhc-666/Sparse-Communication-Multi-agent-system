from __future__ import annotations

import time
import asyncio
from string import Template
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from agentverse.llms.base import BaseChatModel
from agentverse.parser import OutputParser
from agentverse.message import Message, StructuredPrompt
from agentverse.utils import AgentFinish

@dataclass
class SolverResult:
    """LLM Solver result container"""
    answer: str
    reasoning: str
    latency_ms: int
    model: str
    raw_output: str
    success: bool = True
    error_message: Optional[str] = None

class LLMSolverAgent:
    """
    Lightweight LLM Solver Agent for logic problem solving.
    Does not inherit from BaseAgent for better performance and flexibility.
    """
    
    def __init__(
        self,
        name: str,
        display_name: str,
        llm: BaseChatModel,
        output_parser: OutputParser,
        prompt_template: str,
        role_description: str = "",
        instruction: str = "",
        max_retry: int = 3
    ):
        self.name = name
        self.display_name = display_name
        self.llm = llm
        self.output_parser = output_parser
        self.prompt_template = prompt_template
        self.role_description = role_description
        self.instruction = instruction
        self.max_retry = max_retry
        
        # Problem-specific variables
        self.context: str = ""
        self.question: str = ""
        self.options: str = ""
    
    def _fill_prompt_template(self) -> str:
        """Fill the placeholders in the prompt template"""
        
        input_arguments = {
            "agent_name": self.name,
            "role_description": self.role_description,
            "instruction": self.instruction,
            "context": self.context,
            "question": self.question,
            "options": self.options,
        }
        
        return Template(self.prompt_template).safe_substitute(input_arguments)
    
    def _create_structured_prompt(self) -> StructuredPrompt:
        filled_prompt = self._fill_prompt_template()
        
        return StructuredPrompt(
            user_content=filled_prompt
        )
    
    def solve(self, context: str, question: str, options: str) -> SolverResult:
        """
        Solve a logic problem synchronously
        
        Args:
            context: Problem context/background
            question: Question to answer
            options: List of multiple choice options
            
        Returns:
            SolverResult containing answer, reasoning, and metadata
        """
        # Set problem variables
        self.context = context
        self.question = question
        self.options = options
        
        start_time = time.time()
        
        try:
            # Create structured prompt
            structured_prompt = self._create_structured_prompt()
            
            # Generate response with retry logic
            parsed_response = None
            last_error = None
            
            for attempt in range(self.max_retry):
                try:
                    # Call LLM
                    llm_result = self.llm.generate_response(structured_prompt, [])
                    
                    # Parse response
                    parsed_response = self.output_parser.parse(llm_result)
                    break
                    
                except Exception as e:
                    last_error = e
                    if attempt < self.max_retry - 1:
                        time.sleep(1)  # Brief pause before retry
                    continue
            
            # Calculate latency
            latency_ms = int((time.time() - start_time) * 1000)
            
            if parsed_response is None:
                return SolverResult(
                    answer="",
                    reasoning="",
                    latency_ms=latency_ms,
                    model=getattr(self.llm.args, 'model', 'unknown'),
                    raw_output="",
                    success=False,
                    error_message=f"Failed after {self.max_retry} attempts: {last_error}"
                )
            
            # Extract answer and reasoning from parsed response
            if isinstance(parsed_response, AgentFinish):
                output_dict = parsed_response.return_values
                raw_output = output_dict.get("output", "")
            else:
                # Handle case where parsed_response is directly the content
                raw_output = str(parsed_response)
                output_dict = {}
            
            return SolverResult(
                answer=output_dict.get("answer", ""),
                reasoning=output_dict.get("reasoning", ""),
                latency_ms=latency_ms,
                model=getattr(self.llm.args, 'model', 'unknown'),
                raw_output=raw_output,
                success=True
            )
            
        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            
            return SolverResult(
                answer="",
                reasoning="",
                latency_ms=latency_ms,
                model=getattr(self.llm.args, 'model', 'unknown'),
                raw_output="",
                success=False,
                error_message=str(e)
            )
    
    async def asolve(self, context: str, question: str, options: str) -> SolverResult:
        """
        Solve a logic problem asynchronously
        
        Args:
            context: Problem context/background
            question: Question to answer
            options: List of multiple choice options
            
        Returns:
            SolverResult containing answer, reasoning, and metadata
        """
        self.context = context
        self.question = question
        self.options = options
        
        start_time = time.time()
        
        try:
            structured_prompt = self._create_structured_prompt()
            
            # Generate response with retry logic
            parsed_response = None
            last_error = None
            
            for attempt in range(self.max_retry):
                try:
                    llm_result = await self.llm.agenerate_response(structured_prompt, [])
                    
                    parsed_response = self.output_parser.parse(llm_result)
                    break
                    
                except Exception as e:
                    last_error = e
                    if attempt < self.max_retry - 1:
                        await asyncio.sleep(1)  # Brief pause before retry
                    continue
            
            # Calculate latency
            latency_ms = int((time.time() - start_time) * 1000)
            
            if parsed_response is None:
                return SolverResult(
                    answer="",
                    reasoning="",
                    latency_ms=latency_ms,
                    model=getattr(self.llm.args, 'model', 'unknown'),
                    raw_output="",
                    success=False,
                    error_message=f"Failed after {self.max_retry} attempts: {last_error}"
                )
            
            # Extract answer and reasoning from parsed response
            if isinstance(parsed_response, AgentFinish):
                output_dict = parsed_response.return_values
                raw_output = output_dict.get("output", "")
            else:
                # Handle case where parsed_response is directly the content
                raw_output = str(parsed_response)
                output_dict = {}
            
            return SolverResult(
                answer=output_dict.get("answer", ""),
                reasoning=output_dict.get("reasoning", ""),
                latency_ms=latency_ms,
                model=getattr(self.llm.args, 'model', 'unknown'),
                raw_output=raw_output,
                success=True
            )
            
        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            
            return SolverResult(
                answer="",
                reasoning="",
                latency_ms=latency_ms,
                model=getattr(self.llm.args, 'model', 'unknown'),
                raw_output="",
                success=False,
                error_message=str(e)
            )
    
    def reset(self) -> None:
        """Reset the solver state"""
        self.context = ""
        self.question = ""
        self.options = "" 