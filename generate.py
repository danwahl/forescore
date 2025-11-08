#!/usr/bin/env python3
"""
Generate FSM traversal datasets for training language models.

This script generates random finite state machines using AALpy library
along with input sequences and their corresponding final states.
"""

import argparse
import random
import tempfile
import os
from pathlib import Path
from typing import List, Tuple
from datasets import Dataset
from aalpy.utils import generate_random_dfa


def generate_fsm(num_states: int, alphabet: List[str]):
    """
    Generate a random FSM with specified number of states using AALpy.

    Args:
        num_states: Number of states in the FSM
        alphabet: List of input symbols

    Returns:
        AALpy DFA object
    """
    # AALpy guarantees all states are reachable when compute_prefixes=True
    # We set num_accepting_states to 1 (AALpy requires at least 1)
    # but we ignore accepting states in our traversal problems
    dfa = generate_random_dfa(
        alphabet=alphabet,
        num_states=num_states,
        num_accepting_states=1,
        compute_prefixes=True  # Ensures all states are reachable
    )
    return dfa


def fsm_to_dot(dfa) -> str:
    """
    Convert AALpy DFA to simplified DOT notation for traversal problems.

    We simplify AALpy's DOT format by:
    - Using "Start" instead of "__start0" for the initial state marker
    - Removing accepting state annotations (doublecircle) since we only care about traversal
    - Using consistent state naming

    Args:
        dfa: AALpy DFA object

    Returns:
        DOT notation string
    """
    lines = ["digraph FSM {", "  rankdir=LR;", "  node [shape=circle];", ""]

    # Initial state marker
    lines.append("  Start [shape=point];")
    lines.append(f"  Start -> {dfa.initial_state.state_id};")
    lines.append("")

    # Transitions
    # Group transitions by (src, dst) pair to combine labels
    edge_labels = {}
    for state in dfa.states:
        for symbol, next_state in state.transitions.items():
            key = (state.state_id, next_state.state_id)
            if key not in edge_labels:
                edge_labels[key] = []
            edge_labels[key].append(symbol)

    # Write edges with combined labels
    for (src, dst), symbols in sorted(edge_labels.items()):
        label = ",".join(sorted(symbols))
        lines.append(f'  {src} -> {dst} [label="{label}"];')

    lines.append("}")
    return "\n".join(lines)


def simulate_fsm(dfa, sequence: List[str]) -> Tuple[str, List[str]]:
    """
    Simulate FSM on an input sequence.

    Args:
        dfa: AALpy DFA object
        sequence: List of input symbols

    Returns:
        Tuple of (final_state, trace) where trace is list of states visited
    """
    current_state = dfa.initial_state
    trace = [current_state.state_id]

    for symbol in sequence:
        if symbol not in current_state.transitions:
            # No transition defined - stay in current state
            # This shouldn't happen with AALpy DFAs since they're complete,
            # but we handle it for robustness
            pass
        else:
            current_state = current_state.transitions[symbol]
        trace.append(current_state.state_id)

    return current_state.state_id, trace


def generate_sequence(dfa, length: int) -> List[str]:
    """
    Generate a random input sequence that is valid for the FSM.

    Args:
        dfa: AALpy DFA object
        length: Length of sequence to generate

    Returns:
        List of input symbols
    """
    sequence = []
    current_state = dfa.initial_state
    alphabet = dfa.get_input_alphabet()

    for _ in range(length):
        # Get available transitions from current state
        available_symbols = list(current_state.transitions.keys())

        if not available_symbols:
            # No outgoing transitions - just pick random symbol from alphabet
            symbol = random.choice(alphabet)
        else:
            # Pick a random available transition
            symbol = random.choice(available_symbols)

        sequence.append(symbol)

        # Update current state
        if symbol in current_state.transitions:
            current_state = current_state.transitions[symbol]

    return sequence


def format_problem(dfa, sequence: List[str]) -> str:
    """
    Format the FSM and sequence as a problem statement.

    Args:
        dfa: AALpy DFA object
        sequence: Input sequence

    Returns:
        Formatted problem string
    """
    dot = fsm_to_dot(dfa)
    sequence_str = ", ".join(sequence)

    problem = f"""{dot}

Starting from the initial state, process this sequence of inputs:
{sequence_str}

What is the final state?"""

    return problem


def generate_dataset(
    num_examples: int,
    num_states_range: Tuple[int, int],
    sequence_length_range: Tuple[int, int],
    alphabet: List[str],
    seed: int = 42
) -> Dataset:
    """
    Generate a dataset of FSM traversal problems.

    Args:
        num_examples: Number of examples to generate
        num_states_range: (min, max) number of states
        sequence_length_range: (min, max) sequence length
        alphabet: List of input symbols
        seed: Random seed for reproducibility

    Returns:
        HuggingFace Dataset
    """
    random.seed(seed)

    examples = []

    for i in range(num_examples):
        # Sample complexity parameters
        num_states = random.randint(*num_states_range)
        seq_length = random.randint(*sequence_length_range)

        # Generate FSM using AALpy
        dfa = generate_fsm(num_states, alphabet)

        # Generate sequence
        sequence = generate_sequence(dfa, seq_length)

        # Simulate to get answer
        final_state, trace = simulate_fsm(dfa, sequence)

        # Format problem
        problem = format_problem(dfa, sequence)

        # Create example
        example = {
            "problem": problem,
            "sequence": sequence,
            "final_state": final_state,
            "trace": trace,
            "num_states": num_states,
            "sequence_length": seq_length,
        }

        examples.append(example)

        if (i + 1) % 1000 == 0:
            print(f"Generated {i + 1}/{num_examples} examples...")

    return Dataset.from_list(examples)


def main():
    parser = argparse.ArgumentParser(
        description="Generate FSM traversal datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--num-examples",
        type=int,
        default=10000,
        help="Number of examples to generate"
    )

    parser.add_argument(
        "--min-states",
        type=int,
        default=3,
        help="Minimum number of states"
    )

    parser.add_argument(
        "--max-states",
        type=int,
        default=10,
        help="Maximum number of states"
    )

    parser.add_argument(
        "--min-sequence-length",
        type=int,
        default=5,
        help="Minimum sequence length"
    )

    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=20,
        help="Maximum sequence length"
    )

    parser.add_argument(
        "--alphabet",
        type=str,
        default="a,b",
        help="Comma-separated list of input symbols"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data",
        help="Output directory for datasets"
    )

    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Dataset name (auto-generated if not specified)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    # Parse alphabet
    alphabet = [s.strip() for s in args.alphabet.split(",")]

    # Generate dataset name if not provided
    if args.dataset_name is None:
        dataset_name = f"fsm_s{args.min_states}-{args.max_states}_l{args.min_sequence_length}-{args.max_sequence_length}_n{args.num_examples}"
    else:
        dataset_name = args.dataset_name

    print(f"Generating dataset: {dataset_name}")
    print(f"  Examples: {args.num_examples}")
    print(f"  States: {args.min_states}-{args.max_states}")
    print(f"  Sequence length: {args.min_sequence_length}-{args.max_sequence_length}")
    print(f"  Alphabet: {alphabet}")
    print(f"  Seed: {args.seed}")
    print()

    # Generate dataset
    dataset = generate_dataset(
        num_examples=args.num_examples,
        num_states_range=(args.min_states, args.max_states),
        sequence_length_range=(args.min_sequence_length, args.max_sequence_length),
        alphabet=alphabet,
        seed=args.seed
    )

    # Save dataset
    output_path = Path(args.output_dir) / dataset_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving dataset to {output_path}...")
    dataset.save_to_disk(str(output_path))

    # Print sample
    print("\n" + "="*80)
    print("Sample example:")
    print("="*80)
    sample = dataset[0]
    print(sample["problem"])
    print(f"\nAnswer: {sample['final_state']}")
    print(f"Trace: {' → '.join(sample['trace'])}")
    print("="*80)

    print(f"\n✓ Dataset saved successfully!")
    print(f"  Location: {output_path}")
    print(f"  Total examples: {len(dataset)}")


if __name__ == "__main__":
    main()
