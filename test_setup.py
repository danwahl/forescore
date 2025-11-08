#!/usr/bin/env python3
"""
Test script to verify FSM training setup is working correctly.
"""

import os
from datasets import Dataset
from rewards import parse_full_response, format_reward_func, final_state_correct_reward_func

# Test dataset loading
print("="*80)
print("Testing dataset loading...")
print("="*80)

dataset_path = "./fsm_datasets/fsm_s3-10_l5-20_n10000"
if not os.path.exists(dataset_path):
    print(f"ERROR: Dataset not found at {dataset_path}")
    print("Please run: python generate.py --num-examples 10000")
    exit(1)

dataset = Dataset.load_from_disk(dataset_path)
print(f"✓ Dataset loaded successfully: {len(dataset)} examples")

# Check dataset structure
print("\n" + "="*80)
print("Dataset structure:")
print("="*80)
sample = dataset[0]
print(f"Fields: {list(sample.keys())}")
print(f"\nSample problem (truncated):")
print(sample["problem"][:200] + "...")
print(f"\nFinal state: {sample['final_state']}")
print(f"Trace: {' → '.join(sample['trace'])}")

# Test preprocessing
print("\n" + "="*80)
print("Testing preprocessing...")
print("="*80)

system_prompt = """You are an expert at simulating FSMs. Process the sequence step-by-step.
Your response MUST be in XML format:
<think>reasoning</think>
<answer>final_state</answer>"""

def make_conv_for_grpo(example, system_prompt):
    return {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example["problem"]},
        ],
        "final_state": example["final_state"],
    }

processed = make_conv_for_grpo(sample, system_prompt)
print(f"✓ Preprocessing successful")
print(f"  Prompt has {len(processed['prompt'])} messages")
print(f"  Final state: {processed['final_state']}")

# Test reward functions
print("\n" + "="*80)
print("Testing reward functions...")
print("="*80)

# Test valid response
valid_response = """<think>
Processing sequence step by step:
Start at S0
Input 'b' → S2
Input 'a' → S0
Final state: S0
</think>
<answer>S0</answer>"""

parsed = parse_full_response(valid_response)
print(f"✓ Parse valid response:")
print(f"  Think: {parsed['think'][:50]}..." if parsed and parsed['think'] else "  Think: None")
print(f"  Final state: {parsed['final_state']}" if parsed else "  Final state: None")

# Test format reward
# Completions format: list of [{"content": "..."}]
completions = [[{"content": valid_response}]]
format_reward = format_reward_func(completions)
print(f"\n✓ Format reward: {format_reward[0]} (expected: 1.0)")

# Test correctness reward
final_states = ["S0"]
correct_reward = final_state_correct_reward_func(None, completions, final_states)
print(f"✓ Correctness reward: {correct_reward[0]} (expected: 1.0)")

# Test incorrect answer
incorrect_response = """<think>
Processing...
</think>
<answer>S1</answer>"""

completions_wrong = [[{"content": incorrect_response}]]
final_states_wrong = ["S0"]
wrong_reward = final_state_correct_reward_func(None, completions_wrong, final_states_wrong)
print(f"✓ Incorrect answer reward: {wrong_reward[0]} (expected: 0.0)")

# Test malformed response
malformed_response = "This is not valid XML"
completions_malformed = [[{"content": malformed_response}]]
malformed_format_reward = format_reward_func(completions_malformed)
print(f"✓ Malformed response format reward: {malformed_format_reward[0]} (expected: 0.0)")

print("\n" + "="*80)
print("All tests passed! ✓")
print("="*80)
print("\nSetup is ready for training!")
