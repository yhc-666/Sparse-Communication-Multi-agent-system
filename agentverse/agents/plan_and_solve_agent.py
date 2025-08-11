from __future__ import annotations

import time
import asyncio
import re
from string import Template
from typing import Dict, Any, Optional, List

from agentverse.agents.llm_solver_agent import LLMSolverAgent, SolverResult
from agentverse.message import StructuredPrompt
from agentverse.utils import AgentFinish


class PlanAndSolveAgent(LLMSolverAgent):
    """
    Plan-and-Solve Agent that implements the two-step reasoning approach:
    1. First call: Generate detailed reasoning plan and execute it
    2. Second call: Extract the final answer from the reasoning
    
    Based on the Plan-and-Solve-Prompting methodology.
    """
    
    def __init__(
        self,
        name: str,
        display_name: str,
        llm,
        output_parser,
        prompt_template: str,
        role_description: str = "",
        instruction: str = "",
        max_retry: int = 3,
        extraction_trigger: str = "Therefore, the answer is"
    ):
        super().__init__(name, display_name, llm, output_parser, prompt_template, role_description, instruction, max_retry)
        self.extraction_trigger = extraction_trigger
    
    def _create_planning_prompt(self) -> StructuredPrompt:
        """Create the first prompt for planning and reasoning"""
        input_arguments = {
            "context": self.context,
            "question": self.question,
            "options": self.options,
            "instruction": self.instruction,
        }
        
        filled_prompt = Template(self.prompt_template).safe_substitute(input_arguments)
        
        return StructuredPrompt(user_content=filled_prompt)
    
    def _create_extraction_prompt(self, reasoning: str) -> StructuredPrompt:
        """Create the second prompt for answer extraction"""
        # Format similar to the original Plan-and-Solve project
        extraction_prompt = f"Q: {self.question}\n\nOptions:\n{self.options}\n\n{reasoning}\n\n{self.extraction_trigger}"
        
        return StructuredPrompt(user_content=extraction_prompt)
    
    
    def solve(self, context: str, question: str, options: str) -> SolverResult:
        """
        Solve a logic problem using the Plan-and-Solve two-step approach
        """
        # Set problem variables
        self.context = context
        self.question = question
        self.options = options
        
        start_time = time.time()
        
        try:
            # Step 1: Planning and reasoning
            planning_prompt = self._create_planning_prompt()
            
            reasoning_response = None
            last_error = None
            
            for attempt in range(self.max_retry):
                try:
                    # First LLM call for reasoning
                    reasoning_result = self.llm.generate_response(planning_prompt, [])
                    reasoning_response = reasoning_result.content
                    break
                    
                except Exception as e:
                    last_error = e
                    if attempt < self.max_retry - 1:
                        time.sleep(1)
                    continue
            
            if reasoning_response is None:
                latency_ms = int((time.time() - start_time) * 1000)
                return SolverResult(
                    answer="",
                    reasoning="",
                    latency_ms=latency_ms,
                    model=getattr(self.llm.args, 'model', 'unknown'),
                    raw_output="",
                    success=False,
                    error_message=f"Failed reasoning step after {self.max_retry} attempts: {last_error}"
                )
            
            # Try to parse directly from reasoning using output parser
            from agentverse.parser import LLMResult
            llm_result = LLMResult(
                content=reasoning_response,
                send_tokens=0,  # Dummy values for manual parsing
                recv_tokens=0,
                total_tokens=0
            )
            parsed_result = self.output_parser.parse(llm_result)
            
            if isinstance(parsed_result, AgentFinish):
                output_dict = parsed_result.return_values
                if output_dict.get("answer"):
                    latency_ms = int((time.time() - start_time) * 1000)
                    return SolverResult(
                        answer=output_dict.get("answer", ""),
                        reasoning=output_dict.get("reasoning", reasoning_response),
                        latency_ms=latency_ms,
                        model=getattr(self.llm.args, 'model', 'unknown'),
                        raw_output=reasoning_response,
                        success=True
                    )
            
            # Step 2: Answer extraction (if needed)
            extraction_prompt = self._create_extraction_prompt(reasoning_response)
            
            answer_response = None
            for attempt in range(self.max_retry):
                try:
                    # Second LLM call for answer extraction
                    answer_result = self.llm.generate_response(extraction_prompt, [])
                    answer_response = answer_result.content
                    break
                    
                except Exception as e:
                    last_error = e
                    if attempt < self.max_retry - 1:
                        time.sleep(1)
                    continue
            
            if answer_response is None:
                latency_ms = int((time.time() - start_time) * 1000)
                return SolverResult(
                    answer="",
                    reasoning=reasoning_response,
                    latency_ms=latency_ms,
                    model=getattr(self.llm.args, 'model', 'unknown'),
                    raw_output=reasoning_response,
                    success=False,
                    error_message=f"Failed answer extraction after {self.max_retry} attempts: {last_error}"
                )
            
            # Parse the complete response using output parser
            from agentverse.parser import LLMResult
            complete_response = f"{reasoning_response}\n\n{answer_response}"
            llm_result = LLMResult(
                content=complete_response,
                send_tokens=0,  # Dummy values since we're combining two responses
                recv_tokens=0,
                total_tokens=0
            )
            parsed_result = self.output_parser.parse(llm_result)
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            if isinstance(parsed_result, AgentFinish):
                output_dict = parsed_result.return_values
                return SolverResult(
                    answer=output_dict.get("answer", ""),
                    reasoning=output_dict.get("reasoning", reasoning_response),
                    latency_ms=latency_ms,
                    model=getattr(self.llm.args, 'model', 'unknown'),
                    raw_output=complete_response,
                    success=True
                )
            else:
                return SolverResult(
                    answer="",
                    reasoning=reasoning_response,
                    latency_ms=latency_ms,
                    model=getattr(self.llm.args, 'model', 'unknown'),
                    raw_output=complete_response,
                    success=False,
                    error_message="Failed to parse response"
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
        Solve a logic problem asynchronously using the Plan-and-Solve two-step approach
        """
        self.context = context
        self.question = question
        self.options = options
        
        start_time = time.time()
        
        try:
            # Step 1: Planning and reasoning
            planning_prompt = self._create_planning_prompt()
            
            reasoning_response = None
            last_error = None
            
            for attempt in range(self.max_retry):
                try:
                    # First LLM call for reasoning
                    reasoning_result = await self.llm.agenerate_response(planning_prompt, [])
                    reasoning_response = reasoning_result.content
                    break
                    
                except Exception as e:
                    last_error = e
                    if attempt < self.max_retry - 1:
                        await asyncio.sleep(1)
                    continue
            
            if reasoning_response is None:
                latency_ms = int((time.time() - start_time) * 1000)
                return SolverResult(
                    answer="",
                    reasoning="",
                    latency_ms=latency_ms,
                    model=getattr(self.llm.args, 'model', 'unknown'),
                    raw_output="",
                    success=False,
                    error_message=f"Failed reasoning step after {self.max_retry} attempts: {last_error}"
                )
            
            # # Try to parse directly from reasoning using output parser
            # from agentverse.parser import LLMResult
            # llm_result = LLMResult(
            #     content=reasoning_response,
            #     send_tokens=0,  # Dummy values for manual parsing
            #     recv_tokens=0,
            #     total_tokens=0
            # )
            # parsed_result = self.output_parser.parse(llm_result)
            
            # if isinstance(parsed_result, AgentFinish):
            #     output_dict = parsed_result.return_values
            #     if output_dict.get("answer"):
            #         latency_ms = int((time.time() - start_time) * 1000)
            #         return SolverResult(
            #             answer=output_dict.get("answer", ""),
            #             reasoning=reasoning_response,
            #             latency_ms=latency_ms,
            #             model=getattr(self.llm.args, 'model', 'unknown'),
            #             raw_output=reasoning_response,
            #             success=True
            #         )
            
            # Step 2: Answer extraction
            extraction_prompt = self._create_extraction_prompt(reasoning_response)
            
            answer_response = None
            for attempt in range(self.max_retry):
                try:
                    # Second LLM call for answer extraction
                    answer_result = await self.llm.agenerate_response(extraction_prompt, [])
                    answer_response = answer_result.content
                    break
                    
                except Exception as e:
                    last_error = e
                    if attempt < self.max_retry - 1:
                        await asyncio.sleep(1)
                    continue
            
            if answer_response is None:
                latency_ms = int((time.time() - start_time) * 1000)
                return SolverResult(
                    answer="",
                    reasoning=reasoning_response,
                    latency_ms=latency_ms,
                    model=getattr(self.llm.args, 'model', 'unknown'),
                    raw_output=reasoning_response,
                    success=False,
                    error_message=f"Failed answer extraction after {self.max_retry} attempts: {last_error}"
                )
            
            # Parse the complete response using output parser
            from agentverse.parser import LLMResult
            llm_result = LLMResult(
                content=answer_response,
                send_tokens=0,  # Dummy values since we're combining two responses
                recv_tokens=0,
                total_tokens=0
            )
            parsed_result = self.output_parser.parse(llm_result)
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            if isinstance(parsed_result, AgentFinish):
                output_dict = parsed_result.return_values
                return SolverResult(
                    answer=output_dict.get("answer", ""),
                    reasoning=reasoning_response,
                    latency_ms=latency_ms,
                    model=getattr(self.llm.args, 'model', 'unknown'),
                    raw_output=answer_response,
                    success=True
                )
            else:
                return SolverResult(
                    answer="",
                    reasoning=reasoning_response,
                    latency_ms=latency_ms,
                    model=getattr(self.llm.args, 'model', 'unknown'),
                    raw_output=answer_response,
                    success=False,
                    error_message="Failed to parse response"
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