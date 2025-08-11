#!/usr/bin/env python3
"""
Main entry point for running symbolic solver.
Usage: python run_symbolicsolver.py
"""

import yaml
import sys
import os

# Add solver_engine/src to path
solver_src_path = os.path.join(os.path.dirname(__file__), 'solver_engine', 'src')
sys.path.insert(0, solver_src_path)


from logic_inference import LogicInferenceEngine


def main():
    """Main function to run symbolic solver with fixed configuration."""
    # Load configuration from fixed path
    config_path = 'solver_engine/symbolic_solver_config.yaml'
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML configuration: {e}")
        sys.exit(1)
    
    # Extract configuration values
    paths = config.get('paths', {})
    solver = config.get('solver', {})
    processing = config.get('processing', {})
    dataset_config = config.get('dataset', {})
    
    # Create arguments object
    class Args:
        def __init__(self):
            self.input_file = paths.get('input_file', 'sample_data/new_sample_input.json')
            self.output_file = paths.get('output_file', 'sample_data/new_sample_output.json')
            self.backup_LLM_result_path = paths.get('backup_LLM_result_path', '')
            self.backup_strategy = solver.get('backup_strategy', 'random')
            self.cleanup_cache = processing.get('cleanup_cache', True)
            self.verbose = processing.get('verbose', False)
            self.dataset_name = dataset_config.get('name', 'ProofWriter')  # Default to ProofWriter
    
    args = Args()
    
    try:
        engine = LogicInferenceEngine(args)
        stats = engine.inference_on_dataset()
        print(f"Successfully completed inference. Results saved to: {args.output_file}")
        
        # Print statistics in table format
        print("\nStatistics Summary:")
        print("=" * 75)
        print(f"{'Solver':<8} {'Success':<10} {'Parsing Error':<15} {'Execution Error':<17} {'Total':<8}")
        print("-" * 75)
        
        for solver in ['LP', 'FOL', 'SAT']:
            success = stats[solver]['success']
            parsing_error = stats[solver]['parsing error']
            execution_error = stats[solver]['execution error']
            total = stats['total']
            
            print(f"{solver:<8} {success:<10} {parsing_error:<15} {execution_error:<17} {total:<8}")
        
        print("-" * 75)
        
    except Exception as e:
        print(f"Error during inference: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
