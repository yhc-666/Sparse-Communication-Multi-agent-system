#!/usr/bin/env python3
"""
Evaluation script that performs majority voting without final debate.
Takes in results from LLM solvers and symbolic solvers, performs majority voting,
and computes accuracy against gold answers.
"""

import json
import random
from collections import Counter
from typing import Dict, List, Tuple

# Configuration - modify these paths as needed
LLM_SOLVER_PATH = "outputs/deepseek/ProofWriter/llm_solver/llm_solver_results.json"
SYMBOLIC_SOLVER_PATH = "outputs/deepseek/ProofWriter/symbolic_solver/results.json"
output_path = "outputs/deepseek/ProofWriter/evaluation_wo_finaldebate.json"


def load_json_file(filepath: str) -> List[Dict]:
    """Load JSON file and return the data."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found - {filepath}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file - {filepath}")
        return []


def extract_solver_predictions(llm_data: List[Dict], symbolic_data: List[Dict]) -> Dict[str, Dict[str, str]]:
    """
    Extract predictions from all solvers for each question ID.
    Only includes questions that exist in llm_data (which have gold answers).
    Returns: {question_id: {solver_name: prediction}}
    """
    predictions = {}
    valid_question_ids = set()
    
    # First, get all question IDs from LLM data (these have gold answers)
    for item in llm_data:
        question_id = item.get("id", "")
        if question_id:
            valid_question_ids.add(question_id)
    
    # Process LLM solver data (COT Solver and Plan-and-Solve)
    for item in llm_data:
        question_id = item.get("id", "")
        if question_id not in predictions:
            predictions[question_id] = {}
            
        roles = item.get("roles", {})
        for solver_name, solver_data in roles.items():
            if solver_name in ["COT Solver", "Plan-and-Solve"]:
                prediction = solver_data.get("predict", "")
                if prediction:  # Only add if prediction exists
                    predictions[question_id][solver_name] = prediction
    
    # Process symbolic solver data (LP, FOL, SAT) - only for questions in llm_data
    for item in symbolic_data:
        question_id = item.get("id", "")
        # Skip if this question is not in the LLM data
        if question_id not in valid_question_ids:
            continue
            
        if question_id not in predictions:
            predictions[question_id] = {}
            
        roles = item.get("roles", {})
        for solver_name, solver_data in roles.items():
            if solver_name in ["LP", "FOL", "SAT"]:
                # Include all predictions regardless of status
                prediction = solver_data.get("predict", "")
                if prediction:  # Only add if prediction exists
                    predictions[question_id][solver_name] = prediction
    
    return predictions


def majority_voting(predictions: Dict[str, str]) -> str:
    """
    Perform majority voting on predictions.
    If there's a tie, randomly select from the tied options.
    """
    if not predictions:
        return "C"  # Default to Unknown if no predictions
    
    # Count votes
    vote_counts = Counter(predictions.values())
    
    # Get the maximum vote count
    max_votes = max(vote_counts.values())
    
    # Get all options with maximum votes (handles ties)
    tied_options = [option for option, count in vote_counts.items() if count == max_votes]
    
    # Randomly select from tied options
    return random.choice(tied_options)


def compute_accuracy(predictions: Dict[str, str], gold_answers: Dict[str, str]) -> Tuple[float, int, int]:
    """
    Compute accuracy of predictions against gold answers.
    Returns: (accuracy, correct_count, total_count)
    """
    correct = 0
    total = 0
    
    for question_id, prediction in predictions.items():
        if question_id in gold_answers:
            total += 1
            if prediction == gold_answers[question_id]:
                correct += 1
    
    accuracy = (correct / total * 100) if total > 0 else 0
    return accuracy, correct, total


def main():
    print("=" * 60)
    print("Evaluation Without Final Debate - Majority Voting")
    print("=" * 60)
    
    # Load data files
    print(f"\nLoading LLM solver results from: {LLM_SOLVER_PATH}")
    llm_data = load_json_file(LLM_SOLVER_PATH)
    
    print(f"Loading symbolic solver results from: {SYMBOLIC_SOLVER_PATH}")
    symbolic_data = load_json_file(SYMBOLIC_SOLVER_PATH)
    
    if not llm_data or not symbolic_data:
        print("\nError: Failed to load data files. Exiting.")
        return
    
    print(f"\nLoaded {len(llm_data)} questions from LLM solver")
    print(f"Loaded {len(symbolic_data)} questions from symbolic solver")
    
    # Extract gold answers from LLM data
    gold_answers = {}
    for item in llm_data:
        question_id = item.get("id", "")
        gold_answer = item.get("gold_answer", "")
        if question_id and gold_answer:
            gold_answers[question_id] = gold_answer
    
    print(f"\nFound {len(gold_answers)} questions with gold answers")
    
    # Extract predictions from all solvers
    all_predictions = extract_solver_predictions(llm_data, symbolic_data)
    
    # Perform majority voting for each question
    final_predictions = {}
    solver_participation = Counter()
    
    print("\n" + "=" * 60)
    print("Performing Majority Voting")
    print("=" * 60)
    
    for question_id, solver_predictions in all_predictions.items():
        # Track which solvers participated
        for solver in solver_predictions.keys():
            solver_participation[solver] += 1
        
        # Perform majority voting
        final_prediction = majority_voting(solver_predictions)
        final_predictions[question_id] = final_prediction
        
        # Debug output for first few questions
        if len(final_predictions) <= 3:
            print(f"\nQuestion: {question_id}")
            print(f"Solver predictions: {solver_predictions}")
            print(f"Majority vote result: {final_prediction}")
            if question_id in gold_answers:
                print(f"Gold answer: {gold_answers[question_id]}")
    
    # Compute accuracy
    accuracy, correct, total = compute_accuracy(final_predictions, gold_answers)
    
    # Display results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print("\nSolver Participation:")
    for solver, count in sorted(solver_participation.items()):
        percentage = (count / total * 100) if total > 0 else 0
        print(f"  {solver}: {count}/{total} ({percentage:.1f}%)")
    
    print(f"\nTotal questions evaluated: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Incorrect predictions: {total - correct}")
    print(f"\nAccuracy: {accuracy:.2f}%")
    
    # Save detailed results
    output_data = {
        "summary": {
            "total_questions": total,
            "correct_predictions": correct,
            "accuracy": accuracy,
            "solver_participation": dict(solver_participation)
        },
        "predictions": []
    }
    
    for question_id in sorted(final_predictions.keys()):
        output_data["predictions"].append({
            "id": question_id,
            "solver_predictions": all_predictions.get(question_id, {}),
            "majority_vote": final_predictions[question_id],
            "gold_answer": gold_answers.get(question_id, ""),
            "correct": final_predictions[question_id] == gold_answers.get(question_id, "")
        })
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()