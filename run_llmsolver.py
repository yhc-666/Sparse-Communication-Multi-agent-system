import os

# 移除硬编码的环境变量设置，现在通过YAML配置文件处理
# os.environ["OPENAI_API_KEY"] = "sk-733e47bc35da4b49b0bc7ca99ede48f8"
# os.environ["OPENAI_BASE_URL"] = "https://api.deepseek.com/v1"

# always remember to put these lines at the top of your code if you are using clash
# os.environ["http_proxy"] = "http://127.0.0.1:7890"
# os.environ["https_proxy"] = "http://127.0.0.1:7890"
# os.environ["all_proxy"] = "socks5://127.0.0.1:7890"

import json
import asyncio
import time
from typing import Dict, List, Any
from argparse import ArgumentParser
from tqdm import tqdm

from agentverse.agents.llm_solver_agent import LLMSolverAgent, SolverResult
from agentverse.agents.plan_and_solve_agent import PlanAndSolveAgent
from agentverse.agentverse import setup_file_logging
# Import to ensure parsers are registered
from agentverse.tasks.llm_as_solver.output_parser import COTParser, PlanAndSolveParser

parser = ArgumentParser()
parser.add_argument("--config", type=str, default="agentverse/tasks/llm_as_solver/llm_solver_config.yaml")

args = parser.parse_args()

print(args)

import yaml
with open(args.config, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

args_data_path = config['data_path']
args_output_dir = config['output_dir']

os.makedirs(args_output_dir, exist_ok=True)
with open(os.path.join(args_output_dir, "args.txt"), "w") as f:
    f.writelines(str(args))

with open(args_data_path) as f:
    data = json.load(f)

max_questions = config.get('execution', {}).get('max_questions', 0)
if max_questions > 0:
    data = data[:max_questions]

from agentverse.initialization import load_llm
from agentverse.parser import output_parser_registry

llm_config = config['llm_config']
roles = config['roles']

# 处理API凭证配置（与agentverse/initialization.py中的逻辑保持一致）
mode = llm_config.get('mode', 'api')
if mode == "api" and "api_credentials" in llm_config:
    credentials = llm_config["api_credentials"]
    if "openai_api_key" in credentials:
        os.environ["OPENAI_API_KEY"] = credentials["openai_api_key"]
        print(f"🔑 Set OPENAI_API_KEY from config")
    if "openai_base_url" in credentials:
        os.environ["OPENAI_BASE_URL"] = credentials["openai_base_url"]
        print(f"🌐 Set OPENAI_BASE_URL to: {credentials['openai_base_url']}")

if mode == 'api':
    llm_settings = llm_config['api_settings']
else:
    llm_settings = llm_config['local_settings']

llm = load_llm(llm_settings)

setup_file_logging(args_output_dir)

solver_agents = []
for role_config in roles:
    # Get parser type for this role
    parser_type = role_config.get('parser_type', 'cot')  # Default to COT parser
    output_parser = output_parser_registry.build(parser_type)
    
    # Determine agent type based on role name or explicit agent_type field
    agent_type = role_config.get('agent_type', 'llm_solver')
    
    # If agent_type is not explicitly set, infer from role name
    if agent_type == 'llm_solver' and 'plan-and-solve' in role_config['name'].lower():
        agent_type = 'plan_and_solve'
    
    if agent_type == 'plan_and_solve':
        agent = PlanAndSolveAgent(
            name=role_config['name'],
            display_name=role_config['display_name'],
            llm=llm,
            output_parser=output_parser,
            prompt_template=role_config['prompt_template'],
            role_description=role_config['role_description'],
            instruction=role_config['instruction'],
            max_retry=llm_settings.get('max_retry', 3),
            extraction_trigger=role_config.get('extraction_trigger', "Therefore, the answer is")
        )
    else:
        agent = LLMSolverAgent(
            name=role_config['name'],
            display_name=role_config['display_name'],
            llm=llm,
            output_parser=output_parser,
            prompt_template=role_config['prompt_template'],
            role_description=role_config['role_description'],
            instruction=role_config['instruction'],
            max_retry=llm_settings.get('max_retry', 3)
        )
    
    solver_agents.append(agent)

print(f"Initialized {len(solver_agents)} solver agents")

async def solve_question_async(
    question_id: str, 
    context: str, 
    question: str, 
    options: str, 
    gold_answer: str, 
    agents: List[LLMSolverAgent]
) -> Dict[str, Any]:
    """Solve a single question with all agents concurrently"""
    # Prepare result structure
    result = {
        'id': question_id,
        'context': context,
        'question': question,
        'options': options,
        'gold_answer': gold_answer,
        'roles': {}
    }
    
    # Run all agents concurrently
    tasks = []
    for agent in agents:
        task = asyncio.create_task(agent.asolve(context, question, options))
        tasks.append((agent, task))
    
    # Wait for all tasks to complete
    for agent, task in tasks:
        try:
            solver_result = await task
            result['roles'][agent.display_name] = {
                'predict': solver_result.answer,
                'reasoning': solver_result.reasoning,
                'latency_ms': solver_result.latency_ms,
                'model': solver_result.model
            }
            if not solver_result.success:
                result['roles'][agent.display_name]['error'] = solver_result.error_message
        except Exception as e:
            print(f"Error in {agent.display_name}: {e}")
            result['roles'][agent.display_name] = {
                'predict': '',
                'reasoning': '',
                'latency_ms': 0,
                'model': agent.llm.args.model if hasattr(agent.llm, 'args') else 'unknown',
                'error': str(e)
            }
    
    return result

def solve_question_sync(
    question_id: str, 
    context: str, 
    question: str, 
    options: str, 
    gold_answer: str, 
    agents: List[LLMSolverAgent]
) -> Dict[str, Any]:
    """Solve a single question with all agents synchronously"""
    # Prepare result structure
    result = {
        'id': question_id,
        'context': context,
        'question': question,
        'options': options,
        'gold_answer': gold_answer,
        'roles': {}
    }
    
    # Run agents sequentially in sync mode
    for agent in agents:
        try:
            solver_result = agent.solve(context, question, options)
            result['roles'][agent.display_name] = {
                'predict': solver_result.answer,
                'reasoning': solver_result.reasoning,
                'latency_ms': solver_result.latency_ms,
                'model': solver_result.model
            }
            if not solver_result.success:
                result['roles'][agent.display_name]['error'] = solver_result.error_message
        except Exception as e:
            print(f"Error in {agent.display_name}: {e}")
            result['roles'][agent.display_name] = {
                'predict': '', 
                'reasoning': '',
                'latency_ms': 0,
                'model': agent.llm.args.model if hasattr(agent.llm, 'args') else 'unknown',
                'error': str(e)
            }
    
    return result

async def main_async():
    """Main async execution"""
    llm_solver_output = []
    start_time = time.time()
    
    for num, ins in enumerate(tqdm(data, desc="Processing questions")):
        print(f"================================instance {num}====================================")
        
        # preprocess question data (TODO: 设计一个统一的preprocess class做dataset respective处理)
        if "ProofWriter" in args_data_path:
            question_id = ins.get('id', f'Q{num:05d}')
            context = ins.get('context', '')
            question = ins.get('question', '')
            options = '\n'.join([opt.strip() for opt in ins.get('options', [])])
            gold_answer = ins.get('answer', '')

        elif "ProntoQA" in args_data_path:
            pass

        elif "smoketest" in args_data_path:
            question_id = ins.get('id', f'Q{num:05d}')
            context = ins.get('context', '')
            question = ins.get('question', '')
            options = '\n'.join([opt.strip() for opt in ins.get('options', [])])
            gold_answer = ins.get('answer', '')
        
        result = await solve_question_async(
            question_id, context, question, options, gold_answer, solver_agents
        )
        llm_solver_output.append(result)
        
        # Save results immediately after each question
        with open(os.path.join(args_output_dir, "llm_solver_results.json"), "w") as f:
            json.dump(llm_solver_output, f, indent=2, ensure_ascii=False)
    
    # Save final results
    os.makedirs(args_output_dir, exist_ok=True)
    with open(os.path.join(args_output_dir, "llm_solver_results.json"), "w") as f:
        json.dump(llm_solver_output, f, indent=2, ensure_ascii=False)
    
    total_time = time.time() - start_time
    print(f"Completed processing {len(llm_solver_output)} questions in {total_time:.2f} seconds")

def main_sync():
    """Main sync execution"""
    llm_solver_output = []
    start_time = time.time()
    
    for num, ins in enumerate(tqdm(data, desc="Processing questions")):
        print(f"================================instance {num}====================================")
        
        # preprocess question data (TODO: 设计一个统一的preprocess class做dataset respective处理)
        if "ProofWriter" in args_data_path:
            question_id = ins.get('id', f'Q{num:05d}')
            context = ins.get('context', '')
            question = ins.get('question', '')
            options = '\n'.join([opt.strip() for opt in ins.get['options']])
            gold_answer = ins.get('answer', '')

        elif "ProntoQA" in args_data_path:
            pass
        
        elif "smoketest" in args_data_path:
            question_id = ins.get('id', f'Q{num:05d}')
            context = ins.get('context', '')
            question = ins.get('question', '')
            options = '\n'.join([opt.strip() for opt in ins.get('options', [])])
            gold_answer = ins.get('answer', '')
        
        result = solve_question_sync(
            question_id, context, question, options, gold_answer, solver_agents
        )
        llm_solver_output.append(result)
        
        # Save results immediately after each question
        with open(os.path.join(args_output_dir, "llm_solver_results.json"), "w") as f:
            json.dump(llm_solver_output, f, indent=2, ensure_ascii=False)
    
    # Save final results
    os.makedirs(args_output_dir, exist_ok=True)
    with open(os.path.join(args_output_dir, "llm_solver_results.json"), "w") as f:
        json.dump(llm_solver_output, f, indent=2, ensure_ascii=False)
    
    total_time = time.time() - start_time
    print(f"Completed processing {len(llm_solver_output)} questions in {total_time:.2f} seconds")



enable_concurrency = config.get('execution', {}).get('enable_role_concurrency', True)
if enable_concurrency:
    asyncio.run(main_async())
else:
    main_sync()
