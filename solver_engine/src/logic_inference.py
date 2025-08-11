import argparse
import json
import os
import yaml
from tqdm import tqdm

from symbolic_solvers.pyke_solver.pyke_solver import Pyke_Program
from symbolic_solvers.fol_solver.prover9_solver import FOL_Prover9_Program
from symbolic_solvers.z3_solver.sat_problem_solver import LSAT_Z3_Program
from backup_answer_generation import Backup_Answer_Generator



# Simple class mapping for solver types
SOLVER_CLASSES = {
    'LP': Pyke_Program,
    'FOL': FOL_Prover9_Program,
    'SAT': LSAT_Z3_Program,
}


class LogicInferenceEngine:
    def __init__(self, args):
        self.args = args
        self.dataset = self.load_logic_programs(args.input_file)
        self.output_file = args.output_file
        
        # Get dataset name from args
        self.dataset_name = args.dataset_name

        # optional, use LLMwCOT & random guess as backup answers in case solver fails
        self.backup_strategy = args.backup_strategy
        self.backup_LLM_result_path = args.backup_LLM_result_path
        
        # Create backup generators for all solvers using the same dataset name
        self.backup_generators = {
            'LP': Backup_Answer_Generator(self.dataset_name, self.backup_strategy, self.backup_LLM_result_path),
            'FOL': Backup_Answer_Generator(self.dataset_name, self.backup_strategy, self.backup_LLM_result_path),
            'SAT': Backup_Answer_Generator(self.dataset_name, self.backup_strategy, self.backup_LLM_result_path),
        }

    def load_logic_programs(self, input_file):
        with open(input_file, 'r') as f:
            dataset = json.load(f)
        print(f"Loaded {len(dataset)} examples from {input_file}")
        return dataset

    def save_results(self, outputs):
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        with open(self.output_file, 'w') as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)

    def safe_execute_program(self, key, logic_program, example_id):
        """
        returns:
            answer: the answer to the question
            status_code: 'success' or 'execution error' or 'parsing error'
            error_message: the error message if status_code is 'execution error' or 'parsing error'
            reasoning: the reasoning if status_code is 'success'
        """
        solver_class = SOLVER_CLASSES[key]
        program = solver_class(logic_program, self.dataset_name)

        if not getattr(program, 'flag', True): # flag 表示是否成功parse逻辑程序SL, 不代表execute成功与否
            answer = self.backup_generators[key].get_backup_answer(example_id)
            return answer, 'parsing error', '', ''

        answer, err, reasoning = program.execute_program()
        if answer is None:
            answer = self.backup_generators[key].get_backup_answer(example_id)
            # 确保错误信息为字符串，避免JSON序列化问题
            err_str = str(err) if err is not None else ''
            return answer, 'execution error', err_str, ''

        # answer 为各solver原始输出, 需要通过answer_mapping映射为option
        mapped = program.answer_mapping(answer)
        
        # 修正空 reasoning 的状态码 - 如果 reasoning 为空，说明是执行错误
        status_code = 'success'
        error_message = ''
        if reasoning == '' and key in ['LP', 'FOL']:
            status_code = 'execution error'
            error_message = 'Empty reasoning indicates execution failure'
        
        return mapped, status_code, error_message, reasoning

    def inference_on_dataset(self):
        outputs = []
        # Initialize statistics counters
        stats = {
            'LP': {'success': 0, 'parsing error': 0, 'execution error': 0},
            'FOL': {'success': 0, 'parsing error': 0, 'execution error': 0},
            'SAT': {'success': 0, 'parsing error': 0, 'execution error': 0},
            'total': 0
        }
        
        for example in tqdm(self.dataset, desc="Processing logic inference"):
            result = {
                'id': example.get('id'),
                'context': example.get('context'),
                'question': example.get('question'),
                'options': example.get('options'),  # 改为 options
                'answer': example.get('answer'),
                'roles': {
                    'LP': {
                        'predict': '',
                        'reasoning': '',
                        'status_code': '',
                        'error_message': ''
                    },
                    'FOL': {
                        'predict': '',
                        'reasoning': '',
                        'status_code': '',
                        'error_message': ''
                    },
                    'SAT': {
                        'predict': '',
                        'reasoning': '',
                        'status_code': '',
                        'error_message': ''
                    }
                }
            }
            
            # 从translation字段获取逻辑程序
            translation = example.get('translation', [{}])[0]
            
            for key in ['LP', 'FOL', 'SAT']:
                logic_str = translation.get(key, '')
                if logic_str:  # 只有当逻辑程序存在时才执行
                    predicted, status_code, err, reasoning = self.safe_execute_program(key, logic_str, example['id'])
                    result['roles'][key]['predict'] = predicted
                    result['roles'][key]['status_code'] = status_code
                    result['roles'][key]['error_message'] = err
                    result['roles'][key]['reasoning'] = reasoning
                    
                else:
                    # 如果逻辑程序为空，设置默认值
                    result['roles'][key]['predict'] = ''
                    result['roles'][key]['status_code'] = 'parsing error'
                    result['roles'][key]['error_message'] = 'Empty logic program'
                    result['roles'][key]['reasoning'] = ''
                
                # Update statistics
                stats[key][result['roles'][key]['status_code']] += 1
            
            outputs.append(result)
            stats['total'] += 1

        self.save_results(outputs)
        
        # Return statistics
        return stats

def load_config_file(config_path):
    """
    Load configuration from YAML file
    
    Args:
        config_path (str): Path to YAML configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found.")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML configuration: {e}")
        return None


def create_args_from_config(config):
    """
    Create arguments object from configuration dictionary
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        argparse.Namespace: Arguments object
    """
    # Extract paths
    paths = config.get('paths', {})
    input_file = paths.get('input_file', os.path.join('sample_data', 'new_sample_input.json'))
    output_file = paths.get('output_file', os.path.join('sample_data', 'new_sample_output.json'))
    backup_LLM_result_path = paths.get('backup_LLM_result_path', '')
    
    # Extract solver configuration
    solver = config.get('solver', {})
    backup_strategy = solver.get('backup_strategy', 'random')
    
    # Extract dataset configuration
    dataset_config = config.get('dataset', {})
    dataset_name = dataset_config.get('name', 'ProofWriter')
    
    # Create argparse.Namespace object
    args = argparse.Namespace(
        input_file=input_file,
        output_file=output_file,
        backup_LLM_result_path=backup_LLM_result_path,
        backup_strategy=backup_strategy,
        dataset_name=dataset_name
    )
    
    return args


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='outputs/deepseek/ProofWriter/translation/translation_results.json')
    parser.add_argument('--output_file', type=str, default='outputs/deepseek/ProofWriter/symbolic_solver/results.json')
    parser.add_argument('--backup_strategy', type=str, default='random', choices=['random', 'LLM'])
    parser.add_argument('--backup_LLM_result_path', type=str, default='') # path to the LLMwCOT result
    parser.add_argument('--dataset_name', type=str, default='ProofWriter', help='Dataset name')
    parser.add_argument('--config', type=str, default='', help='Path to YAML configuration file')
    
    args = parser.parse_args()
    
    # If config file is provided, load it and merge with command line arguments
    if args.config and os.path.exists(args.config):
        config = load_config_file(args.config)
        if config:
            config_args = create_args_from_config(config)
            # Command line arguments override config file
            for key, value in vars(args).items():
                if key != 'config' and value == parser.get_default(key):
                    # Use config file value if command line uses default
                    setattr(args, key, getattr(config_args, key, value))
    
    return args


if __name__ == '__main__':
    args = parse_args()
    engine = LogicInferenceEngine(args)
    engine.inference_on_dataset()
