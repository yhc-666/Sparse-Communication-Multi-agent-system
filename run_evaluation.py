import json
import os
import argparse
from tqdm import tqdm
import re
from collections import Counter

def normalize_answer(answer):
    """
    Normalize answer format to handle variations like 'A) True' vs 'A'
    """
    if not answer:
        return ""
    
    # Convert to string and strip whitespace
    answer = str(answer).strip()
    
    # Extract the option letter (A, B, C, etc.)
    match = re.match(r'^([A-Z])', answer.upper())
    if match:
        return match.group(1)
    
    # If no letter found, return the original answer in uppercase
    return answer.upper()


def get_last_speaker(chat_history):
    """
    Get the role of the last speaker from chat history
    """
    if not chat_history:
        return None
    
    # Find the last message that contains actual content (not just answer tags)
    for message in reversed(chat_history):
        if message.get('content') and not message['content'].startswith('<answer>'):
            return message['role']
    
    # If all messages are answer tags, return the last one
    return chat_history[-1]['role']


def get_majority_vote_prediction(final_predictions):
    """
    Get prediction using majority vote from all supporters
    """
    if not final_predictions:
        return None, {}
    
    # Collect all valid predictions
    predictions = []
    for role, prediction_data in final_predictions.items():
        if isinstance(prediction_data, dict) and 'predict' in prediction_data:
            predict = prediction_data['predict']
            normalized_predict = normalize_answer(predict)
            if normalized_predict:
                predictions.append(normalized_predict)
    
    if not predictions:
        return None, {}
    
    # Count votes
    vote_counts = Counter(predictions)
    
    # Get the majority vote (most common answer)
    most_common = vote_counts.most_common(1)
    majority_answer = most_common[0][0] if most_common else None
    
    # Create stats dictionary
    stats = {
        'total_voters': len(predictions),
        'vote_counts': dict(vote_counts),
        'majority_answer': majority_answer,
        'majority_count': most_common[0][1] if most_common else 0
    }
    
    return majority_answer, stats


def get_aggregated_prediction(item, strategy):
    """
    Get aggregated prediction based on strategy
    """
    final_predictions = item.get('Final predictions', {})
    
    if strategy == 'last_speaker':
        # Get the last speaker from chat history
        chat_history = item.get('chat_history', [])
        last_speaker = get_last_speaker(chat_history)
        
        if not last_speaker:
            return None, {'error': 'No valid speaker found'}
        
        if last_speaker not in final_predictions:
            return None, {'error': f'No prediction found for speaker {last_speaker}'}
        
        predicted_answer = final_predictions[last_speaker].get('predict', '')
        normalized_predicted = normalize_answer(predicted_answer)
        
        return normalized_predicted, {'strategy': 'last_speaker', 'speaker': last_speaker}
    
    elif strategy == 'majority_vote':
        predicted_answer, stats = get_majority_vote_prediction(final_predictions)
        stats['strategy'] = 'majority_vote'
        return predicted_answer, stats
    
    else:
        return None, {'error': f'Unknown strategy: {strategy}'}


def evaluate_final_debate(json_path, aggregation_strategy='last_speaker'):
    """
    Evaluate the final debate results and calculate accuracy
    """
    # Check if file exists
    if not os.path.exists(json_path):
        print(f"Error: File not found at {json_path}")
        return None
    
    # Read JSON file
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    correct_count = 0
    total_count = len(data)
    error_count = 0
    tie_count = 0  # For majority vote ties
    
    print(f"Starting evaluation of {total_count} questions using '{aggregation_strategy}' strategy...")
    
    for item in tqdm(data, desc="Evaluating"):
        try:
            # Get gold answer
            gold_answer = item.get('gold_answer', '')
            normalized_gold = normalize_answer(gold_answer)
            
            # Get aggregated prediction
            predicted_answer, stats = get_aggregated_prediction(item, aggregation_strategy)
            
            if predicted_answer is None:
                if 'error' in stats:
                    print(f"Warning: {stats['error']} for question {item.get('id', 'unknown')}")
                error_count += 1
                continue
            
            # Track ties for majority vote
            if aggregation_strategy == 'majority_vote' and 'majority_count' in stats:
                total_voters = int(stats.get('total_voters', 0))
                majority_count = int(stats.get('majority_count', 0))
                if total_voters > 1 and majority_count <= total_voters // 2:
                    tie_count += 1
            
            # Check if answers match
            if normalized_gold == predicted_answer:
                correct_count += 1
            
        except Exception as e:
            print(f"Error processing question {item.get('id', 'unknown')}: {e}")
            error_count += 1
    
    # Calculate accuracy
    valid_questions = total_count - error_count
    if valid_questions == 0:
        print("Error: No valid questions found")
        return None
    
    accuracy = correct_count / valid_questions
    
    print(f"\n{'='*50}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"Aggregation strategy: {aggregation_strategy}")
    print(f"Total questions: {total_count}")
    print(f"Valid questions: {valid_questions}")
    print(f"Correct answers: {correct_count}")
    print(f"Errors encountered: {error_count}")
    if aggregation_strategy == 'majority_vote' and tie_count > 0:
        print(f"Tie cases (no clear majority): {tie_count}")
    print(f"Overall accuracy: {accuracy:.2%}")
    print(f"{'='*50}")
    
    return accuracy


def main():
    parser = argparse.ArgumentParser(description='Evaluate final debate results')
    parser.add_argument(
        '--input_path', 
        type=str,
        default='outputs/deepseek-chat/ProofWriter/final_debate/train/final_debate_results.json',
        help='Path to the final debate results JSON file'
    )
    parser.add_argument(
        '--aggregation_strategy',
        type=str,
        choices=['last_speaker', 'majority_vote'],
        default='majority_vote',
        help='Strategy for aggregating predictions: last_speaker (default) or majority_vote'
    )
    
    args = parser.parse_args()
    
    accuracy = evaluate_final_debate(args.input_path, args.aggregation_strategy)
    
    if accuracy is not None:
        print(f"\nEvaluation completed successfully!")
    else:
        print(f"\nEvaluation failed!")
        exit(1)

if __name__ == "__main__":
    main()
