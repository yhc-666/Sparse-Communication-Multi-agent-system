import os
from typing import Dict, List

import yaml
# from bmtools.agent.singletool import import_all_apis, load_single_tools
from langchain.agents import Agent as langchainAgent

# from langchain.chat_models import ChatOpenAI
# from langchain.chat_models.base import BaseChatModel
# from langchain.llms import OpenAI
# from langchain.llms.base import BaseLLM
from agentverse.llms import OpenAIChat, llm_registry  # OpenAICompletion commented out as unused

# from langchain.memory import ChatMessageHistory
from langchain.memory.prompt import _DEFAULT_SUMMARIZER_TEMPLATE
from langchain.prompts import PromptTemplate

# from agentverse.agents import Agent
from agentverse.agents import agent_registry
from agentverse.environments import BaseEnvironment, env_registry
from agentverse.memory import memory_registry
from agentverse.memory_manipulator import memory_manipulator_registry

# from agentverse.memory.memory import SummaryMemory
from agentverse.parser import output_parser_registry


def load_llm(llm_config: Dict):
    llm_type = llm_config.pop("llm_type", "gpt-3.5-turbo")  # Changed default from text-davinci-003 to gpt-3.5-turbo

    return llm_registry.build(llm_type, **llm_config)


def load_memory(memory_config: Dict):
    memory_type = memory_config.pop("memory_type", "chat_history")
    return memory_registry.build(memory_type, **memory_config)


def load_memory_manipulator(memory_manipulator_config: Dict):
    memory_manipulator_type = memory_manipulator_config.pop("memory_manipulator_type", "basic")
    return memory_manipulator_registry.build(memory_manipulator_type, **memory_manipulator_config)


def load_tools(tool_config: List[Dict]):
    if len(tool_config) == 0:
        return []
    all_tools_list = []
    # for tool in tool_config:
    #     _, config = load_single_tools(tool["tool_name"], tool["tool_url"])
    #     all_tools_list += import_all_apis(config)
    return all_tools_list


def load_environment(env_config: Dict) -> BaseEnvironment:
    env_type = env_config.pop("env_type", "basic")
    return env_registry.build(env_type, **env_config)


def load_agent(agent_config: Dict) -> langchainAgent:
    agent_type = agent_config.pop("agent_type", "conversation")
    agent = agent_registry.build(agent_type, **agent_config)
    return agent


def prepare_task_config(taskwithyaml):
    """Read the yaml config of the given task in `tasks` directory."""

    if not str(taskwithyaml).endswith("config.yaml"):
        raise ValueError(
            "You should include config.yaml in your task config path"
        )

    if not os.path.exists(taskwithyaml):
        raise ValueError(
            "You should include the config.yaml file in the task directory"
        )
    task_config = yaml.safe_load(open(taskwithyaml))
    task = task_config["task"]
    # Build the output parser
    dataset_name = task_config.get('dataset', {}).get('name', 'ProofWriter')
    parser = output_parser_registry.build(task, dataset_name=dataset_name)
    task_config["output_parser"] = parser

    # Handle llm_config if present
    selected_llm_config = None
    if "llm_config" in task_config:
        llm_config = task_config["llm_config"]
        mode = llm_config.get("mode", "api")
        
        # 处理API凭证配置
        if mode == "api" and "api_credentials" in llm_config:
            credentials = llm_config["api_credentials"]
            if "openai_api_key" in credentials:
                os.environ["OPENAI_API_KEY"] = credentials["openai_api_key"]
                print(f"🔑 Set OPENAI_API_KEY from config")
            if "openai_base_url" in credentials:
                os.environ["OPENAI_BASE_URL"] = credentials["openai_base_url"]
                print(f"🌐 Set OPENAI_BASE_URL to: {credentials['openai_base_url']}")
        
        if mode == "api":
            selected_llm_config = llm_config.get("api_settings", {})
            print(f"🔗 Using API mode with model: {selected_llm_config.get('model', 'default')}")
        elif mode == "local":
            selected_llm_config = llm_config.get("local_settings", {})
            print(f"🏠 Using Local LLM mode with model: {selected_llm_config.get('model_path', 'default')}")
            # Update output_dir to reflect local mode
            if "output_dir" in task_config:
                task_config["output_dir"] = task_config["output_dir"].replace("deepseek_chat", "deepseek_local")
        else:
            raise ValueError(f"Invalid llm_config mode: {mode}. Must be 'api' or 'local'")

    for i, agent_configs in enumerate(task_config["agents"]):
        agent_configs["memory"] = load_memory(agent_configs.get("memory", {}))
        memory_manipulator = load_memory_manipulator(agent_configs.get("memory_manipulator", {}))
        agent_configs["memory_manipulator"] = memory_manipulator
        if agent_configs.get("tool_memory", None) is not None:
            agent_configs["tool_memory"] = load_memory(agent_configs["tool_memory"])
        
        # Use selected_llm_config if available, otherwise use agent's own llm config
        if selected_llm_config:
            llm_config_to_use = selected_llm_config.copy()
        else:
            llm_config_to_use = agent_configs.get("llm", {"llm_type": "gpt-3.5-turbo"})  # Changed default from text-davinci-003 to gpt-3.5-turbo
            
        llm = load_llm(llm_config_to_use)
        agent_configs["llm"] = llm

        # Extract max_retry from LLM config and pass it to agent
        if "max_retry" in llm_config_to_use:
            agent_configs["max_retry"] = llm_config_to_use["max_retry"]

        agent_configs["tools"] = load_tools(agent_configs.get("tools", []))

        agent_configs["output_parser"] = task_config["output_parser"]

    return task_config
