import logging
import numpy as np
import time
import os
import json
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field

from agentverse.llms.base import LLMResult

from . import llm_registry
from .base import BaseChatModel, BaseCompletionModel, BaseModelArgs
from agentverse.message import Message, StructuredPrompt

logger = logging.getLogger(__name__)
# 创建专门用于记录输入输出的logger
io_logger = logging.getLogger(f"{__name__}.io")

try:
    import openai
    from openai import OpenAI, AsyncOpenAI
    from openai import OpenAIError
    
    # 延迟初始化客户端，避免在模块导入时就要求环境变量
    client = None
    aclient = None
    
    def get_client():
        global client
        if client is None:
            client = OpenAI()  # 自动从环境变量读取 OPENAI_API_KEY 和 OPENAI_BASE_URL
        return client
    
    def get_async_client():
        global aclient
        if aclient is None:
            aclient = AsyncOpenAI()  # 自动从环境变量读取
        return aclient
    
    is_openai_available = True
except ImportError:
    is_openai_available = False
    logging.warning("openai package is not installed")
    client = None
    aclient = None
    
    def get_client():
        return None
    
    def get_async_client():
        return None


class OpenAIChatArgs(BaseModelArgs):
    model: str = Field(default="gpt-3.5-turbo")
    max_tokens: int = Field(default=2048)
    temperature: float = Field(default=1.0)
    top_p: float = Field(default=1.0)
    n: int = Field(default=1)
    stop: Optional[Union[str, List]] = Field(default=None)
    presence_penalty: int = Field(default=0)
    frequency_penalty: int = Field(default=0)


# 注释掉的 OpenAICompletion 相关代码
# class OpenAICompletionArgs(OpenAIChatArgs):
#     model: str = Field(default="text-davinci-003")
#     suffix: str = Field(default="")
#     best_of: int = Field(default=1)


# @llm_registry.register("text-davinci-003")
# class OpenAICompletion(BaseCompletionModel):
#     args: OpenAICompletionArgs = Field(default_factory=OpenAICompletionArgs)

#     def __init__(self, max_retry: int = 3, **kwargs):
#         args = OpenAICompletionArgs()
#         args = args.dict()
#         for k, v in args.items():
#             args[k] = kwargs.pop(k, v)
#         if len(kwargs) > 0:
#             logging.warning(f"Unused arguments: {kwargs}")
#         super().__init__(args=args, max_retry=max_retry)

#     def generate_response(self, prompt: str, chat_memory: List[Message], final_prompt: str) -> LLMResult:
#         response = client.completions.create(prompt=prompt, **self.args.dict())
#         return LLMResult(
#             content=response.choices[0].text,
#             send_tokens=response.usage.prompt_tokens,
#             recv_tokens=response.usage.completion_tokens,
#             total_tokens=response.usage.total_tokens,
#         )

#     async def agenerate_response(self, prompt: str, chat_memory: List[Message], final_prompt: str) -> LLMResult:
#         response = await aclient.completions.create(prompt=prompt, **self.args.dict())
#         return LLMResult(
#             content=response.choices[0].text,
#             send_tokens=response.usage.prompt_tokens,
#             recv_tokens=response.usage.completion_tokens,
#             total_tokens=response.usage.total_tokens,
#         )

@llm_registry.register("gpt-3.5-turbo-0301")
@llm_registry.register("gpt-3.5-turbo")
@llm_registry.register("gpt-4")
@llm_registry.register("deepseek-chat") # added by xiang
@llm_registry.register("gpt-4.1")
class OpenAIChat(BaseChatModel):
    args: OpenAIChatArgs = Field(default_factory=OpenAIChatArgs)

    def __init__(self, max_retry: int = 3, **kwargs):
        args = OpenAIChatArgs()
        args = args.dict()

        for k, v in args.items():
            args[k] = kwargs.pop(k, v)
        if len(kwargs) > 0:
            logging.warning(f"Unused arguments: {kwargs}")
        super().__init__(args=args, max_retry=max_retry)

    def _construct_messages(self, structured_prompt: "StructuredPrompt", chat_memory: List[Message]):
        """
        创建messages json, inclusing 3 sections:
        - system message: 来自structured_prompt.system_content
        - assistant messages with name: 来自chat_memory
        - user message: 来自structured_prompt.user_content
        """
        
        messages = []
        
        # 1. 添加system message
        if structured_prompt.system_content:
            messages.append({
                "role": "system",
                "content": structured_prompt.system_content
            })
        
        # 2. 添加历史对话作为assistant messages
        for message in chat_memory:
            if message.content.strip() and message.content != "[Silence]":
                messages.append({
                    "role": "assistant", 
                    "name": message.sender.replace(" ", "_"),  # 确保name符合API要求
                    "content": message.content
                })
        
        # 3. 添加用户消息
        if structured_prompt.user_content:
            messages.append({
                "role": "user",
                "content": structured_prompt.user_content
            })
        
        return messages

    def generate_response(self, structured_prompt: "StructuredPrompt", chat_memory: List[Message]) -> LLMResult:
        messages = self._construct_messages(structured_prompt, chat_memory)
        try:
            response = get_client().chat.completions.create(messages=messages, **self.args.dict())
        except (OpenAIError, KeyboardInterrupt) as error:
            raise
        return LLMResult(
            content=response.choices[0].message.content,
            send_tokens=response.usage.prompt_tokens,
            recv_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        )

    async def agenerate_response(self, structured_prompt: "StructuredPrompt", chat_memory: List[Message]) -> LLMResult:
        messages = self._construct_messages(structured_prompt, chat_memory)
        
        #io_logger.info("➡️Input Messages JSON:\n%s", json.dumps(messages, ensure_ascii=False, indent=2))
        
        try:
            response = await get_async_client().chat.completions.create(messages=messages, **self.args.dict())
        except (OpenAIError, KeyboardInterrupt) as error:
            raise
        
        result_content = response.choices[0].message.content
        
        # io_logger.info("↩️ LLM Response:\n%s", json.dumps({
        #    "response": result_content
            
        # }, ensure_ascii=False, indent=2))
        
        return LLMResult(
            content=result_content,
            send_tokens=response.usage.prompt_tokens,
            recv_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        )


def get_embedding(text: str, attempts=3) -> np.array:
    attempt = 0
    while attempt < attempts:
        try:
            text = text.replace("\n", " ")
            embedding = get_client().embeddings.create(input=[text], model="text-embedding-ada-002")["data"][0]["embedding"]
            return tuple(embedding)
        except Exception as e:
            attempt += 1
            logger.error(f"Error {e} when requesting openai models. Retrying")
            time.sleep(10)
    logger.warning(
        f"get_embedding() failed after {attempts} attempts. returning empty response"
    )