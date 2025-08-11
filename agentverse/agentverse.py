import asyncio
import logging
import os
from typing import List

# from agentverse.agents import Agent
from agentverse.agents.conversation_agent import BaseAgent
from agentverse.environments import BaseEnvironment
from agentverse.initialization import load_agent, load_environment, prepare_task_config

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

openai_logger = logging.getLogger("openai")
openai_logger.setLevel(logging.WARNING)


# ================================ 配置专门的输入输出日志记录器 ================================
io_logger = logging.getLogger("agentverse.llms.local_llm.io")
io_logger.setLevel(logging.INFO)
io_logger_openai = logging.getLogger("agentverse.llms.openai.io")
io_logger_openai.setLevel(logging.INFO)

def setup_file_logging(output_dir: str):
    """设置文件日志处理器，将JSON输入输出日志保存到文件"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置JSON日志文件路径
    json_log_file = os.path.join(output_dir, "llm_io_logs.txt")
    
    # 创建文件处理器
    file_handler = logging.FileHandler(json_log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 设置文件日志格式（更简洁，专注于JSON内容）
    file_formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    
    # 为两个io logger添加文件处理器
    io_logger.addHandler(file_handler)
    io_logger_openai.addHandler(file_handler)
    
    print(f"📁 JSON日志将保存到: {json_log_file}")
    return json_log_file

# ================================ 配置专门的输入输出日志记录器 ends================================

class AgentVerse:
    def __init__(self, agents: List[BaseAgent], environment: BaseEnvironment):
        self.agents = agents
        self.environment = environment

    @classmethod
    def from_task(cls, task: str):
        """Build an AgentVerse from a task name.
        The task name should correspond to a directory in `tasks` directory.
        Then this method will load the configuration from the yaml file in that directory.
        """
        # Prepare the config of the task

        task_config = prepare_task_config(task)

        # Build the agents
        agents = []
        for agent_configs in task_config["agents"]:
            agent = load_agent(agent_configs)
            agents.append(agent)

        # Build the environment
        env_config = task_config["environment"]
        env_config["agents"] = agents
        environment = load_environment(env_config)

        # Set input_path and output_path
        input_path = task_config["data_path"]
        output_path = task_config["output_dir"]
        
        # 设置文件日志
        setup_file_logging(output_path)

        return cls(agents, environment), input_path, output_path

    def run(self):
        """Run the environment from scratch until it is done."""
        self.environment.reset()
        while not self.environment.is_done():
            asyncio.run(self.environment.step())

    def reset(self):
        self.environment.reset()
        for agent in self.agents:
            agent.reset()

    def next(self, *args, **kwargs):
        """Run the environment for one step and return the return message."""
        return_message = asyncio.run(self.environment.step(*args, **kwargs))
        return return_message
