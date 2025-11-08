#!/usr/bin/env python3
"""
Unit tests for FSM generation using independent parsing and solving logic.

This test module verifies the correctness of FSM generation by:
1. Independently parsing the DOT notation output
2. Independently simulating FSM execution
3. Comparing results with the original implementation
"""

import unittest
import re
import random
from typing import Dict, List, Tuple, Set

from generate import (
    generate_fsm,
    fsm_to_dot,
    simulate_fsm,
    generate_sequence,
    format_problem
)


class IndependentDOTParser:
    """
    Independent DOT notation parser that doesn't rely on the FSM dictionary structure.
    This parser works directly from the DOT string to verify correctness.
    """

    def __init__(self, dot_string: str):
        self.dot_string = dot_string
        self.transitions = {}  # (state, symbol) -> next_state
        self.states = set()
        self.initial_state = None
        self.alphabet = set()
        self._parse()

    def _parse(self):
        """Parse DOT notation to extract FSM structure."""
        lines = self.dot_string.strip().split('\n')

        for line in lines:
            line = line.strip()

            # Find initial state: "Start -> S0;"
            initial_match = re.match(r'Start\s*->\s*(\w+);', line)
            if initial_match:
                self.initial_state = initial_match.group(1)
                self.states.add(self.initial_state)
                continue

            # Find transitions: "S0 -> S1 [label="a"];" or "S0 -> S1 [label="a,b"];"
            transition_match = re.match(r'(\w+)\s*->\s*(\w+)\s*\[label="([^"]+)"\];', line)
            if transition_match:
                src_state = transition_match.group(1)
                dst_state = transition_match.group(2)
                label = transition_match.group(3)

                # Handle comma-separated labels
                symbols = [s.strip() for s in label.split(',')]

                self.states.add(src_state)
                self.states.add(dst_state)

                for symbol in symbols:
                    self.alphabet.add(symbol)
                    key = (src_state, symbol)
                    if key in self.transitions:
                        raise ValueError(f"Duplicate transition: {src_state} --{symbol}--> (already exists)")
                    self.transitions[key] = dst_state

        if self.initial_state is None:
            raise ValueError("No initial state found in DOT notation")

    def get_initial_state(self) -> str:
        """Get the initial state."""
        return self.initial_state

    def get_next_state(self, current_state: str, symbol: str) -> str:
        """
        Get next state for a given (state, symbol) pair.
        If no transition exists, stay in current state.
        """
        key = (current_state, symbol)
        return self.transitions.get(key, current_state)

    def get_all_states(self) -> Set[str]:
        """Get all states in the FSM."""
        return self.states.copy()

    def get_alphabet(self) -> Set[str]:
        """Get the alphabet."""
        return self.alphabet.copy()


class IndependentFSMSimulator:
    """
    Independent FSM simulator that works from parsed DOT notation.
    """

    def __init__(self, parser: IndependentDOTParser):
        self.parser = parser

    def simulate(self, sequence: List[str]) -> Tuple[str, List[str]]:
        """
        Simulate FSM execution on an input sequence.

        Args:
            sequence: List of input symbols

        Returns:
            Tuple of (final_state, trace) where trace is list of states visited
        """
        current_state = self.parser.get_initial_state()
        trace = [current_state]

        for symbol in sequence:
            next_state = self.parser.get_next_state(current_state, symbol)
            current_state = next_state
            trace.append(current_state)

        return current_state, trace


class TestFSMGeneration(unittest.TestCase):
    """Test FSM generation with independent verification."""

    def setUp(self):
        """Set up test fixtures."""
        random.seed(42)

    def test_basic_fsm_generation(self):
        """Test basic FSM generation and DOT conversion."""
        dfa = generate_fsm(num_states=3, alphabet=['a', 'b'])

        # Verify DFA structure (AALpy DFA object)
        self.assertEqual(len(dfa.states), 3)
        self.assertIsNotNone(dfa.initial_state)
        self.assertEqual(set(dfa.get_input_alphabet()), {'a', 'b'})

        # Verify all states have transitions (AALpy DFAs are complete)
        for state in dfa.states:
            self.assertIsNotNone(state.transitions)
            self.assertTrue(len(state.transitions) > 0)

    def test_dot_parsing_consistency(self):
        """Test that DOT notation can be parsed back correctly."""
        dfa = generate_fsm(num_states=5, alphabet=['a', 'b', 'c'])
        dot = fsm_to_dot(dfa)

        # Parse DOT independently
        parser = IndependentDOTParser(dot)

        # Verify initial state matches
        self.assertEqual(parser.get_initial_state(), dfa.initial_state.state_id)

        # Verify all states are present
        parsed_states = parser.get_all_states()
        expected_states = {state.state_id for state in dfa.states}
        self.assertEqual(parsed_states, expected_states)

        # Verify alphabet matches
        expected_alphabet = set()
        for state in dfa.states:
            expected_alphabet.update(state.transitions.keys())
        self.assertEqual(parser.get_alphabet(), expected_alphabet)

    def test_simulation_consistency(self):
        """Test that independent simulation matches original simulation."""
        # Generate several random FSMs and test
        for trial in range(10):
            with self.subTest(trial=trial):
                dfa = generate_fsm(
                    num_states=random.randint(3, 8),
                    alphabet=['a', 'b']
                )

                # Generate a test sequence
                sequence = generate_sequence(dfa, length=random.randint(5, 15))

                # Original simulation
                original_final, original_trace = simulate_fsm(dfa, sequence)

                # Independent simulation
                dot = fsm_to_dot(dfa)
                parser = IndependentDOTParser(dot)
                simulator = IndependentFSMSimulator(parser)
                independent_final, independent_trace = simulator.simulate(sequence)

                # Verify results match
                self.assertEqual(
                    independent_final,
                    original_final,
                    f"Final states don't match for sequence {sequence}\n"
                    f"Original: {original_final}\n"
                    f"Independent: {independent_final}\n"
                    f"DOT:\n{dot}"
                )
                self.assertEqual(
                    independent_trace,
                    original_trace,
                    f"Traces don't match for sequence {sequence}\n"
                    f"Original: {original_trace}\n"
                    f"Independent: {independent_trace}\n"
                    f"DOT:\n{dot}"
                )

    def test_complete_transitions(self):
        """Test that AALpy DFAs are complete (all transitions defined)."""
        # AALpy generates complete DFAs where every state has a transition
        # for every symbol in the alphabet
        dfa = generate_fsm(num_states=5, alphabet=['a', 'b', 'c'])

        # Verify completeness: every state should have transitions for all symbols
        alphabet = dfa.get_input_alphabet()
        for state in dfa.states:
            for symbol in alphabet:
                self.assertIn(
                    symbol,
                    state.transitions,
                    f"State {state.state_id} missing transition for symbol '{symbol}'"
                )
                self.assertIsNotNone(
                    state.transitions[symbol],
                    f"State {state.state_id} has None transition for symbol '{symbol}'"
                )

    def test_single_state_fsm(self):
        """Test FSM with only one state."""
        dfa = generate_fsm(num_states=1, alphabet=['a', 'b'])

        self.assertEqual(len(dfa.states), 1)

        # Generate and test sequence
        sequence = generate_sequence(dfa, length=10)

        original_final, original_trace = simulate_fsm(dfa, sequence)

        dot = fsm_to_dot(dfa)
        parser = IndependentDOTParser(dot)
        simulator = IndependentFSMSimulator(parser)
        independent_final, independent_trace = simulator.simulate(sequence)

        self.assertEqual(independent_final, original_final)
        self.assertEqual(independent_trace, original_trace)

    def test_all_states_reachable(self):
        """Test that all states in generated FSMs are reachable from initial state."""
        # AALpy with compute_prefixes=True guarantees all states are reachable
        for trial in range(20):
            with self.subTest(trial=trial):
                dfa = generate_fsm(
                    num_states=random.randint(3, 10),
                    alphabet=['a', 'b']
                )

                # BFS to find reachable states
                reachable = set()
                queue = [dfa.initial_state]
                reachable.add(dfa.initial_state.state_id)

                while queue:
                    state = queue.pop(0)
                    for symbol in dfa.get_input_alphabet():
                        if symbol in state.transitions:
                            next_state = state.transitions[symbol]
                            if next_state.state_id not in reachable:
                                reachable.add(next_state.state_id)
                                queue.append(next_state)

                # All states should be reachable
                all_state_ids = {state.state_id for state in dfa.states}
                self.assertEqual(
                    reachable,
                    all_state_ids,
                    f"Not all states are reachable. Reachable: {reachable}, All: {all_state_ids}"
                )

    def test_dot_format_validity(self):
        """Test that DOT notation follows correct format."""
        dfa = generate_fsm(num_states=5, alphabet=['a', 'b', 'c'])
        dot = fsm_to_dot(dfa)

        # Check basic structure
        self.assertIn('digraph FSM {', dot)
        self.assertIn('rankdir=LR;', dot)
        self.assertIn('Start [shape=point];', dot)
        self.assertIn(f'Start -> {dfa.initial_state.state_id};', dot)
        self.assertIn('}', dot)

        # Check that all edges have proper format
        edge_pattern = re.compile(r'\s*\w+\s*->\s*\w+\s*\[label="[^"]+"\];')
        lines = dot.split('\n')
        for line in lines:
            if '->' in line and 'Start' not in line:
                self.assertIsNotNone(
                    edge_pattern.match(line),
                    f"Invalid edge format: {line}"
                )

    def test_edge_label_combining(self):
        """Test that DOT format properly handles edge labels."""
        # Generate a DFA and verify DOT format is valid
        dfa = generate_fsm(num_states=3, alphabet=['a', 'b'])
        dot = fsm_to_dot(dfa)

        # Parse the DOT independently
        parser = IndependentDOTParser(dot)

        # Verify that the parsed transitions match the original DFA
        for state in dfa.states:
            for symbol, next_state in state.transitions.items():
                parsed_next = parser.get_next_state(state.state_id, symbol)
                self.assertEqual(
                    parsed_next,
                    next_state.state_id,
                    f"Transition mismatch: {state.state_id} --{symbol}--> "
                    f"expected {next_state.state_id}, got {parsed_next}"
                )

    def test_large_fsm(self):
        """Test larger FSM to ensure scalability."""
        dfa = generate_fsm(num_states=15, alphabet=['a', 'b', 'c', 'd'])
        dot = fsm_to_dot(dfa)

        # Parse and verify
        parser = IndependentDOTParser(dot)
        self.assertEqual(len(parser.get_all_states()), 15)

        # Run simulation test
        sequence = generate_sequence(dfa, length=30)
        original_final, original_trace = simulate_fsm(dfa, sequence)

        simulator = IndependentFSMSimulator(parser)
        independent_final, independent_trace = simulator.simulate(sequence)

        self.assertEqual(independent_final, original_final)
        self.assertEqual(independent_trace, original_trace)

    def test_sequence_generation_validity(self):
        """Test that generated sequences are valid (use alphabet symbols)."""
        dfa = generate_fsm(num_states=5, alphabet=['a', 'b'])

        for _ in range(10):
            sequence = generate_sequence(dfa, length=20)

            # All symbols should be from alphabet
            alphabet = dfa.get_input_alphabet()
            for symbol in sequence:
                self.assertIn(symbol, alphabet)

    def test_format_problem_contains_all_info(self):
        """Test that format_problem includes all necessary information."""
        dfa = generate_fsm(num_states=3, alphabet=['a', 'b'])
        sequence = generate_sequence(dfa, length=10)

        problem = format_problem(dfa, sequence)

        # Check for key components
        self.assertIn('digraph FSM', problem)
        self.assertIn('Start ->', problem)
        self.assertIn(', '.join(sequence), problem)
        self.assertIn('What is the final state?', problem)

    def test_deterministic_with_seed(self):
        """Test that same seed produces same FSM."""
        random.seed(123)
        dfa1 = generate_fsm(num_states=5, alphabet=['a', 'b'])
        dot1 = fsm_to_dot(dfa1)

        random.seed(123)
        dfa2 = generate_fsm(num_states=5, alphabet=['a', 'b'])
        dot2 = fsm_to_dot(dfa2)

        self.assertEqual(dot1, dot2, "Same seed should produce identical FSMs")

    def test_comprehensive_random_tests(self):
        """Comprehensive random testing with various configurations."""
        configs = [
            {'num_states': 3, 'alphabet': ['a', 'b']},
            {'num_states': 7, 'alphabet': ['a', 'b', 'c']},
            {'num_states': 10, 'alphabet': ['x', 'y']},
            {'num_states': 4, 'alphabet': ['0', '1', '2']},
        ]

        for config in configs:
            for trial in range(5):
                with self.subTest(config=config, trial=trial):
                    dfa = generate_fsm(**config)
                    dot = fsm_to_dot(dfa)

                    # Parse independently
                    parser = IndependentDOTParser(dot)
                    simulator = IndependentFSMSimulator(parser)

                    # Test multiple sequences
                    for _ in range(3):
                        sequence = generate_sequence(dfa, length=random.randint(5, 20))

                        original_final, original_trace = simulate_fsm(dfa, sequence)
                        independent_final, independent_trace = simulator.simulate(sequence)

                        self.assertEqual(
                            independent_final,
                            original_final,
                            f"Mismatch with config {config}, sequence {sequence}"
                        )
                        self.assertEqual(
                            independent_trace,
                            original_trace,
                            f"Trace mismatch with config {config}, sequence {sequence}"
                        )


if __name__ == '__main__':
    unittest.main(verbosity=2)
