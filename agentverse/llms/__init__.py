from agentverse.registry import Registry

llm_registry = Registry(name="LLMRegistry")

from .base import BaseLLM, BaseChatModel, BaseCompletionModel, LLMResult
from .openai import OpenAIChat  # OpenAICompletion commented out as unused

# Import local LLM classes
try:
    from .local_llm import BaseLocalLLM, DeepseekLocalLLM, QwenLocalLLM, ChatGLMLocalLLM, LlamaLocalLLM
except ImportError:
    # If vllm is not installed, local LLM classes won't be available
    pass
