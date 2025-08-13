import os
import json
import yaml
from argparse import ArgumentParser
from typing import Dict, List
from tqdm import tqdm
import numpy as np

from agentverse.agentverse import AgentVerse


def load_translation_results(path: str) -> List[Dict]:
    """
    Load FOL translation results from Stage 1.
    
    Args:
        path: Path to translation results JSON file
        
    Returns:
        List of translation instances
    """
    print(f"Loading translation results from: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        translations = json.load(f)
    print(f"Loaded {len(translations)} translation instances")
    return translations


def assign_translation_to_agents(agentverse, instance: Dict) -> None:
    """
    Assign FOL translation and problem data to all agents.
    
    Args:
        agentverse: AgentVerse instance
        instance: Translation instance with context, question, options, and FOL
    """
    # Get FOL translation
    translation = instance.get('translation', {})
    if isinstance(translation, dict):
        fol_translation = translation.get('FOL', '')
    else:
        fol_translation = str(translation)
    
    # Assign to all agents
    for agent in agentverse.agents:
        agent.context = instance.get('context', '')
        agent.question = instance.get('question', '')
        agent.options = '\n'.join(instance.get('options', []))
        # Use FOL translation as initial reasoning
        agent.reasoning = fol_translation
        agent.predict = ""  # Will be updated during debate
        agent.final_prompt = ""
        
    print(f"Assigned translation to {len(agentverse.agents)} agents")


def extract_chat_history(agentverse) -> List[Dict]:
    """
    Extract complete chat history from all agents.
    Since each agent has different memory due to sparse gates,
    we collect from all agents to get the complete picture.
    
    Args:
        agentverse: The AgentVerse instance
        
    Returns:
        List of chat history entries
    """
    chat_history = []
    seen_messages = set()  # To avoid duplicates
    
    # Collect messages from all agents' memories
    for agent in agentverse.agents:
        for message in agent.memory.messages:
            if hasattr(message, 'sender') and hasattr(message, 'content'):
                # Create a unique key for the message
                msg_key = (message.sender, message.content[:100])  # Use first 100 chars as key
                
                if msg_key not in seen_messages:
                    seen_messages.add(msg_key)
                    chat_history.append({
                        "role": message.sender,
                        "content": message.content
                    })
    
    # Sort by some order if needed (could add timestamps if available)
    return chat_history


def collect_final_predictions(agents) -> Dict:
    """
    Collect final predictions from all agents.
    
    Args:
        agents: List of agent instances
        
    Returns:
        Dictionary of final predictions by agent name
    """
    final_predictions = {}
    
    for agent in agents:
        # Use final_answer attribute from the agent
        final_answer = agent.final_answer if hasattr(agent, 'final_answer') else ""
        final_predictions[agent.name] = {
            "predict": final_answer
        }
    
    return final_predictions


def extract_gate_history(visibility_rule) -> Dict:
    """
    Extract gate matrices history from sparse visibility rule.
    
    Args:
        visibility_rule: The sparse visibility rule instance
        
    Returns:
        Dictionary with gate history and statistics
    """
    if not hasattr(visibility_rule, 'gates'):
        return {}
    
    gate_history = {}
    
    for round_num, gates in visibility_rule.gates.items():
        n_agents = gates.shape[0]
        # Convert numpy array to list for JSON serialization
        # Add 1 to round_num for human-readable display (Round 1, Round 2, etc.)
        gate_history[f"round_{round_num + 1}"] = {
            "gates": gates.tolist(),
            "open_gates": int(np.sum(gates) - n_agents),  # Exclude self-connections
            "total_gates": n_agents * (n_agents - 1),
            "sparsity": float(1 - (np.sum(gates) - n_agents) / (n_agents * (n_agents - 1)))
        }
    
    # Add confidence history if available
    if hasattr(visibility_rule, 'confidences'):
        confidence_history = {}
        for round_num, confidences in visibility_rule.confidences.items():
            # Add 1 to round_num for human-readable display
            confidence_history[f"round_{round_num + 1}"] = {
                f"agent_{i}": conf for i, conf in confidences.items()
            }
        gate_history["confidences"] = confidence_history
    
    return gate_history


def main():
    """Main execution function for sparse debate"""
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, 
                       default="agentverse/tasks/final_debate/sparse_debate_config.yaml",
                       help="Path to sparse debate configuration file")
    parser.add_argument("--max_instances", type=int, default=10,
                       help="Maximum number of instances to process (0 for all)")
    parser.add_argument("--start_from", type=int, default=0,
                       help="Start processing from this instance index")
    args = parser.parse_args()
    
    print("=" * 50)
    print("Starting Sparse Communication Debate")
    print("=" * 50)
    
    # Load configuration
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Initialize AgentVerse
    print("\n📋 Initializing AgentVerse with sparse communication...")
    agentverse, _, _ = AgentVerse.from_task(args.config)
    
    # Load translation results
    translation_path = config['translation_results_path']
    translations = load_translation_results(translation_path)
    
    # Determine instances to process
    end_idx = len(translations) if args.max_instances == 0 else min(args.start_from + args.max_instances, len(translations))
    instances_to_process = translations[args.start_from:end_idx]
    
    print(f"\n📊 Processing instances {args.start_from} to {end_idx-1} ({len(instances_to_process)} total)")
    
    # Prepare output
    sparse_debate_output = []
    output_dir = config['output_dir']
    
    # Process each instance
    for num, instance in enumerate(tqdm(instances_to_process, desc="Processing instances", unit="instance")):
        actual_idx = args.start_from + num
        print(f"\n{'='*20} Instance {actual_idx} (ID: {instance.get('id', 'unknown')}) {'='*20}")
        
        try:
            # Assign translation to agents
            assign_translation_to_agents(agentverse, instance)
            
            # Run sparse debate
            print("Running sparse debate...")
            agentverse.run()
            
            # Extract results
            chat_history = extract_chat_history(agentverse)  # Get complete history from all agents
            final_predictions = collect_final_predictions(agentverse.agents)
            
            # Extract gate history from visibility rule
            gate_history = {}
            if hasattr(agentverse.environment.rule, 'visibility'):
                gate_history = extract_gate_history(agentverse.environment.rule.visibility)
            
            # Compile result
            result = {
                "id": instance.get("id", f"instance_{actual_idx}"),
                "context": instance.get("context", ""),
                "question": instance.get("question", ""),
                "options": instance.get("options", []),
                "gold_answer": instance.get("answer", ""),
                "translation": instance.get("translation", {}),
                "chat_history": chat_history,
                "Final predictions": final_predictions,
                "gate_statistics": gate_history
            }
            
            sparse_debate_output.append(result)
            
            # Save intermediate results
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "sparse_debate_results.json"), "w", encoding='utf-8') as f:
                json.dump(sparse_debate_output, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Instance {actual_idx} completed")
            
        except Exception as e:
            print(f"❌ Error processing instance {actual_idx}: {e}")
            import traceback
            traceback.print_exc()
            
            # Add error result
            sparse_debate_output.append({
                "id": instance.get("id", f"instance_{actual_idx}"),
                "error": str(e)
            })
        
        finally:
            # Reset agents for next instance
            for agent in agentverse.agents:
                agent.reset()
            # Reset visibility rule
            if hasattr(agentverse.environment.rule, 'visibility'):
                agentverse.environment.rule.visibility.reset()
    
    # Final summary
    print("\n" + "=" * 50)
    print("SPARSE DEBATE SUMMARY")
    print("=" * 50)
    print(f"Total instances processed: {len(sparse_debate_output)}")
    print(f"Results saved to: {os.path.join(output_dir, 'sparse_debate_results.json')}")
    
    # Calculate average sparsity
    total_sparsity = []
    for result in sparse_debate_output:
        if "gate_statistics" in result:
            for round_key, round_stats in result["gate_statistics"].items():
                if round_key.startswith("round_") and "sparsity" in round_stats:
                    total_sparsity.append(round_stats["sparsity"])
    
    if total_sparsity:
        avg_sparsity = np.mean(total_sparsity)
        print(f"Average sparsity across all rounds: {avg_sparsity:.2%}")
    
    print("\n✅ Sparse debate completed successfully!")
    print(f"You can now run evaluation using:")
    print(f"python run_evaluation.py --input_path {os.path.join(output_dir, 'sparse_debate_results.json')} --aggregation_strategy majority_vote")


if __name__ == "__main__":
    main()