#!/usr/bin/env python3
import json
import sys

def extract_chat_history(json_file, target_id):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for item in data:
        if item['id'] == target_id:
            print(f"ID: {target_id}")
            print("-" * 50)
            for i, message in enumerate(item['chat_history'], 1):
                print(f"{i}. {message['role']}:")
                print(message['content'])
                print("-" * 30)
            return
    
    print(f"ID '{target_id}' not found")

if __name__ == "__main__":
    
    extract_chat_history("outputs/deepseek/ProofWriter/final_debate/final_debate_results.json", "ProofWriter_RelNeg-OWA-D5-81_Q11") 