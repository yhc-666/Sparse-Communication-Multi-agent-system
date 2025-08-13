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
    In regular debate, all agents see all messages.
    
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


def extract_token_usage(agents) -> Dict:
    """
    Extract token usage information from all agents.
    
    Args:
        agents: List of agent instances
        
    Returns:
        Dictionary with token usage statistics
    """
    token_usage = {
        "total_prefilling_tokens": 0,  # Total send tokens (prefilling)
        "total_generation_tokens": 0,  # Total recv tokens (generation)
        "total_all_tokens": 0,
        "per_agent": {},
        "per_round": []
    }
    
    # Collect token usage from each agent
    for agent in agents:
        if hasattr(agent, 'token_usage'):
            agent_usage = agent.token_usage
            token_usage["total_prefilling_tokens"] += agent_usage.get("total_send_tokens", 0)
            token_usage["total_generation_tokens"] += agent_usage.get("total_recv_tokens", 0)
            token_usage["total_all_tokens"] += agent_usage.get("total_tokens", 0)
            
            # Store per-agent breakdown
            token_usage["per_agent"][agent.name] = {
                "send_tokens": agent_usage.get("total_send_tokens", 0),
                "recv_tokens": agent_usage.get("total_recv_tokens", 0),
                "total_tokens": agent_usage.get("total_tokens", 0),
                "rounds": agent_usage.get("rounds", [])
            }
    
    # Aggregate per-round data
    if token_usage["per_agent"]:
        # Get max number of rounds from any agent
        max_rounds = max(len(agent_data["rounds"]) 
                        for agent_data in token_usage["per_agent"].values())
        
        for round_idx in range(max_rounds):
            round_tokens = {
                "round": round_idx + 1,
                "send_tokens": 0,
                "recv_tokens": 0,
                "total_tokens": 0
            }
            
            for agent_data in token_usage["per_agent"].values():
                if round_idx < len(agent_data["rounds"]):
                    round_data = agent_data["rounds"][round_idx]
                    round_tokens["send_tokens"] += round_data.get("send_tokens", 0)
                    round_tokens["recv_tokens"] += round_data.get("recv_tokens", 0)
                    round_tokens["total_tokens"] += round_data.get("total_tokens", 0)
            
            token_usage["per_round"].append(round_tokens)
    
    return token_usage


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


def main():
    """Main execution function for regular debate (without sparse communication)"""
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, 
                       default="agentverse/tasks/final_debate/final_debate_config.yaml",
                       help="Path to regular debate configuration file")
    parser.add_argument("--max_instances", type=int, default=10,
                       help="Maximum number of instances to process (0 for all)")
    parser.add_argument("--start_from", type=int, default=0,
                       help="Start processing from this instance index")
    args = parser.parse_args()
    
    print("=" * 50)
    print("Starting Regular Debate (Full Communication)")
    print("=" * 50)
    
    # Load configuration
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Initialize AgentVerse
    print("\n📋 Initializing AgentVerse with full communication...")
    agentverse, _, _ = AgentVerse.from_task(args.config)
    
    # Load translation results
    translation_path = config['translation_results_path']
    translations = load_translation_results(translation_path)
    
    # Determine instances to process
    end_idx = len(translations) if args.max_instances == 0 else min(args.start_from + args.max_instances, len(translations))
    instances_to_process = translations[args.start_from:end_idx]
    
    print(f"\n📊 Processing instances {args.start_from} to {end_idx-1} ({len(instances_to_process)} total)")
    
    # Prepare output
    regular_debate_output = []
    output_dir = config['output_dir']
    
    # Process each instance
    for num, instance in enumerate(tqdm(instances_to_process, desc="Processing instances", unit="instance")):
        actual_idx = args.start_from + num
        print(f"\n{'='*20} Instance {actual_idx} (ID: {instance.get('id', 'unknown')}) {'='*20}")
        
        try:
            # Assign translation to agents
            assign_translation_to_agents(agentverse, instance)
            
            # Run regular debate
            print("Running regular debate...")
            agentverse.run()
            
            # Extract results
            chat_history = extract_chat_history(agentverse)  # Get complete history from all agents
            final_predictions = collect_final_predictions(agentverse.agents)
            
            # Extract token usage
            token_usage = extract_token_usage(agentverse.agents)
            
            # Compile result (no gate statistics for regular debate)
            result = {
                "id": instance.get("id", f"instance_{actual_idx}"),
                "context": instance.get("context", ""),
                "question": instance.get("question", ""),
                "options": instance.get("options", []),
                "gold_answer": instance.get("answer", ""),
                "translation": instance.get("translation", {}),
                "chat_history": chat_history,
                "Final predictions": final_predictions,
                "token_usage": token_usage
            }
            
            regular_debate_output.append(result)
            
            # Save intermediate results
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "regular_debate_results.json"), "w", encoding='utf-8') as f:
                json.dump(regular_debate_output, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Instance {actual_idx} completed")
            
        except Exception as e:
            print(f"❌ Error processing instance {actual_idx}: {e}")
            import traceback
            traceback.print_exc()
            
            # Add error result
            regular_debate_output.append({
                "id": instance.get("id", f"instance_{actual_idx}"),
                "error": str(e)
            })
        
        finally:
            # Reset agents for next instance
            for agent in agentverse.agents:
                agent.reset()
    
    # Final summary
    print("\n" + "=" * 50)
    print("REGULAR DEBATE SUMMARY")
    print("=" * 50)
    print(f"Total instances processed: {len(regular_debate_output)}")
    print(f"Results saved to: {os.path.join(output_dir, 'regular_debate_results.json')}")
    
    # Calculate average token usage
    total_prefilling_tokens = 0
    
    for result in regular_debate_output:
        # Collect token usage
        if "token_usage" in result:
            total_prefilling_tokens += result["token_usage"].get("total_prefilling_tokens", 0)
    
    if total_prefilling_tokens > 0:
        avg_prefilling_per_question = total_prefilling_tokens / len(regular_debate_output)
        print(f"Total prefilling tokens: {total_prefilling_tokens:,}")
        print(f"Average prefilling tokens per question: {avg_prefilling_per_question:,.0f}")
    
    print("\n✅ Regular debate completed successfully!")
    print(f"You can now run evaluation using:")
    print(f"python run_evaluation.py --input_path {os.path.join(output_dir, 'regular_debate_results.json')} --aggregation_strategy majority_vote")


if __name__ == "__main__":
    main()