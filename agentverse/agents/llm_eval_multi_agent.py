from __future__ import annotations

import logging
import bdb
import re
from string import Template
from typing import TYPE_CHECKING, List

from agentverse.message import Message, StructuredPrompt
from openai import RateLimitError

from . import agent_registry
from .base import BaseAgent

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentverse.environments.base import BaseEnvironment

@agent_registry.register("llm_eval_multi")
class LLMEvalAgent(BaseAgent):

    question: str = ""
    context: str = ""  
    options: str = ""  # for LogicalDeduction dataset options
    # for direct score
    reference_text: str = ""
    generated_text: str = ""

    # for pair comparison
    compared_text_one: str = ""
    compared_text_two: str = ""

    final_prompt: str = ""
    final_prompt_to_use: str = ""
    normal_turn_instruction: str = ""

    def _clean_output(self, text: str) -> str:
        """Clean the output text similar to output_parser.py"""
        cleaned_output = text.strip()
        cleaned_output = re.sub(r"\n+", "\n", cleaned_output)
        cleaned_output = re.sub(r"\n-", "\n", cleaned_output)  # Remove dash after newline
        return cleaned_output

    def step(self, env_description: str = "") -> Message:
        structured_prompt = self._fill_prompt_template(env_description, None)

        parsed_response = None
        response = None
        for i in range(self.max_retry):
            try:
                response = self.llm.generate_response(structured_prompt, self.memory.messages)
                parsed_response = self.output_parser.parse(response, 0, 10, 1, self.name)
                break
            except KeyboardInterrupt:
                raise
            except Exception as e:
                logging.error(e)
                logging.warning("Retrying...")
                continue

        if parsed_response is None:
            logging.error(f"{self.name} failed to generate valid response.")
            # Use cleaned output as fallback when parsing fails
            if response is not None:
                cleaned_content = self._clean_output(response.content)
                logging.info(f"{self.name} using cleaned output as fallback: {cleaned_content[:100]}...")
            else:
                cleaned_content = ""

        message = Message(
            content=cleaned_content
            if parsed_response is None
            else parsed_response.return_values["output"],
            sender=self.name,
            receiver=self.get_receiver(),
        )
        return message

    async def astep(self, env: "BaseEnvironment" = None, env_description: str = "") -> Message:
        """Asynchronous version of step"""

        # TODO modify this line, if it is the final round, add some instruction in the prompt
        # you must use the following format, first give the rate of the summary of the above 4 aspects then finally give the reasoning on why you give this rate
        # Relevance:
        # Consistency:
        # Fluency:
        # Coherence:
        # Thought: (your thought)

        if env.cnt_turn >= env.max_turns - len(env.agents):
            # self.final_prompt = "Now, please give your final judgement, and you must use the following format, first start with 'This is my final judgement!' and briefly give the thought on why you give this rate, then finally give the rate of the summary of the above 4 aspects." \
            #                     "This is my final judgement!\n" \
            #                     "Thought: (your thought)\n" \
            #                     "Relevance:\n" \
            #                     "Consistency:\n" \
            #                     "Fluency:\n" \
            #                     "Coherence:\n" \
            self.final_prompt = self.final_prompt_to_use

        structured_prompt = self._fill_prompt_template(env_description, env)

        parsed_response = None
        response = None

        should_break = False
        while True:

            for i in range(self.max_retry):
                try:
                    response = await self.llm.agenerate_response(structured_prompt, self.memory.messages)
                    parsed_response = self.output_parser.parse(response, env.cnt_turn, env.max_turns, len(env.agents), self.name)
                    should_break = True
                    break
                except (KeyboardInterrupt, bdb.BdbQuit):
                    raise
                except Exception as e:
                    if isinstance(e, RateLimitError):
                        logging.error(e)
                        logging.warning("Retrying Until rate limit error disappear...")
                        break
                    else:
                        logging.error(e)
                        logging.warning(f"Retrying..{i}_th time")
                        continue
            else:
                logging.error(f"After {self.max_retry} failed try, end the loop")
                break
            if should_break:
                break
            else:
                continue

        if parsed_response is None:
            logging.error(f"{self.name} failed to generate valid response.")
            # Use cleaned output as fallback when parsing fails
            if response is not None:
                cleaned_content = self._clean_output(response.content)
                logging.info(f"{self.name} using cleaned output as fallback: {cleaned_content[:100]}...")
            else:
                cleaned_content = ""

        message = Message(
            content=cleaned_content
            if parsed_response is None
            else parsed_response.return_values["output"],
            sender=self.name,
            receiver=self.get_receiver(),
        )
        return message

    def _fill_prompt_template(self, env_description: str = "", env=None) -> StructuredPrompt:
        """Fill the placeholders in the prompt template and return structured prompt components
        
        这个方法将原来的单一prompt字符串拆分为结构化的部分：
        - system_content: ${chat_history}之前的部分
        - user_content: ${chat_history}之后的部分
        """
        # Determine turn-specific instruction based on current turn
        if env and hasattr(env, 'cnt_turn') and hasattr(env, 'max_turns') and hasattr(env, 'agents'):
            if env.cnt_turn < env.max_turns - len(env.agents):
                turn_specific_instruction = Template(self.normal_turn_instruction).safe_substitute(agent_name=self.name)
            else:
                turn_specific_instruction = self.final_prompt
        else:
            # Fallback for when env is not available
            turn_specific_instruction = Template(self.normal_turn_instruction).safe_substitute(agent_name=self.name)
        
        # 填充除了chat_history之外的所有参数
        input_arguments = {
            "agent_name": self.name,
            "env_description": env_description,
            "role_description": self.role_description,
            "question": self.question,
            "context": self.context, 
            "options": self.options,
            "reference_text": self.reference_text,
            "generated_text": self.generated_text,
            "compared_text_one": self.compared_text_one,
            "compared_text_two": self.compared_text_two,
            "final_prompt": self.final_prompt,
            "turn_specific_instruction": turn_specific_instruction,
            # 注意：故意不填充chat_history，保留${chat_history}作为分割点
        }
        
        # 使用safe_substitute，这样未提供的变量（如chat_history）会保留原样
        partially_filled_template = Template(self.prompt_template).safe_substitute(input_arguments)
        
        # 使用${chat_history}作为分割点
        chat_history_placeholder = "${chat_history}"
        
        if chat_history_placeholder in partially_filled_template:
            # 按照${chat_history}分割
            parts = partially_filled_template.split(chat_history_placeholder)
            if len(parts) == 2:
                system_content = parts[0].strip()
                user_content = parts[1].strip()
            else:
                # 如果有多个${chat_history}或其他异常情况，回退到原始逻辑
                system_content = partially_filled_template
                user_content = ""
        else:
            # 如果template中没有${chat_history}，所有内容作为system_content
            system_content = partially_filled_template
            user_content = ""
        
        return StructuredPrompt(
            system_content=system_content,
            user_content=user_content
        )

    def add_message_to_memory(self, messages: List[Message]) -> None:
        self.memory.add_message(messages)

    def reset(self) -> None:
        """Reset the agent"""
        self.memory.reset()
        # TODO: reset receiver
