#!/usr/bin/env python3
"""
Unit tests for reward functions.

Tests verify correct behavior of:
- XML parsing with <think> and <answer> tags
- Format validation rewards
- Final state correctness rewards
- Token count penalty calculation
"""

import unittest
import math

from rewards import (
    parse_full_response,
    format_reward_func,
    final_state_correct_reward_func,
    token_count_reward_func,
    PAD_TOKEN
)


class TestParseFullResponse(unittest.TestCase):
    """Test XML parsing of model responses."""

    def test_valid_response(self):
        """Test parsing valid response with think and answer."""
        text = "<think>Starting at s1, applying a...</think><answer>s3</answer>"
        result = parse_full_response(text)

        self.assertIsNotNone(result)
        self.assertEqual(result["think"], "Starting at s1, applying a...")
        self.assertEqual(result["final_state"], "s3")

    def test_multiline_think(self):
        """Test parsing response with multiline think block."""
        text = """<think>
        Step 1: Start at s1
        Step 2: Apply transition a
        Step 3: Arrive at s2
        </think><answer>s2</answer>"""
        result = parse_full_response(text)

        self.assertIsNotNone(result)
        self.assertIn("Step 1", result["think"])
        self.assertEqual(result["final_state"], "s2")

    def test_whitespace_handling(self):
        """Test parsing with extra whitespace."""
        text = "  <think>  thought  </think>  <answer>  s1  </answer>  "
        result = parse_full_response(text)

        self.assertIsNotNone(result)
        self.assertEqual(result["think"], "  thought  ")
        self.assertEqual(result["final_state"], "s1")

    def test_with_padding_tokens(self):
        """Test parsing with padding tokens."""
        text = f"{PAD_TOKEN}<think>reasoning</think><answer>s5</answer>{PAD_TOKEN}"
        result = parse_full_response(text)

        self.assertIsNotNone(result)
        self.assertEqual(result["think"], "reasoning")
        self.assertEqual(result["final_state"], "s5")

    def test_missing_think(self):
        """Test response with missing think block."""
        text = "<answer>s2</answer>"
        result = parse_full_response(text)

        self.assertIsNotNone(result)
        self.assertIsNone(result["think"])
        self.assertEqual(result["final_state"], "s2")

    def test_missing_answer(self):
        """Test response with missing answer block."""
        text = "<think>some reasoning</think>"
        result = parse_full_response(text)

        self.assertIsNotNone(result)
        self.assertEqual(result["think"], "some reasoning")
        self.assertIsNone(result["final_state"])

    def test_empty_think(self):
        """Test response with empty think block."""
        text = "<think></think><answer>s1</answer>"
        result = parse_full_response(text)

        self.assertIsNotNone(result)
        self.assertIsNone(result["think"])
        self.assertEqual(result["final_state"], "s1")

    def test_empty_answer(self):
        """Test response with empty answer block."""
        text = "<think>thought</think><answer></answer>"
        result = parse_full_response(text)

        self.assertIsNotNone(result)
        self.assertEqual(result["think"], "thought")
        self.assertIsNone(result["final_state"])

    def test_whitespace_only_think(self):
        """Test response with whitespace-only think block."""
        text = "<think>   </think><answer>s3</answer>"
        result = parse_full_response(text)

        self.assertIsNotNone(result)
        self.assertIsNotNone(result["think"])
        self.assertEqual(result["final_state"], "s3")

    def test_malformed_xml(self):
        """Test malformed XML returns None."""
        text = "<think>unclosed tag"
        result = parse_full_response(text)

        self.assertIsNone(result)

    def test_no_xml_tags(self):
        """Test plain text without XML tags returns dict with None values."""
        text = "Just some plain text answer"
        result = parse_full_response(text)

        self.assertIsNotNone(result)
        self.assertIsNone(result["think"])
        self.assertIsNone(result["final_state"])

    def test_wrong_tags(self):
        """Test response with wrong tag names."""
        text = "<reasoning>thought</reasoning><result>s1</result>"
        result = parse_full_response(text)

        self.assertIsNotNone(result)
        self.assertIsNone(result["think"])
        self.assertIsNone(result["final_state"])

    def test_nested_tags(self):
        """Test response with nested tags."""
        text = "<think>outer <inner>nested</inner></think><answer>s2</answer>"
        result = parse_full_response(text)

        self.assertIsNotNone(result)
        self.assertIn("outer", result["think"])
        self.assertEqual(result["final_state"], "s2")


class TestFormatRewardFunc(unittest.TestCase):
    """Test format validation reward function."""

    def test_valid_format(self):
        """Test valid format gets full reward."""
        completions = [[{"content": "<think>reasoning</think><answer>s1</answer>"}]]
        rewards = format_reward_func(completions)

        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)

    def test_multiple_valid_responses(self):
        """Test multiple valid responses."""
        completions = [
            [{"content": "<think>a</think><answer>s1</answer>"}],
            [{"content": "<think>b</think><answer>s2</answer>"}],
            [{"content": "<think>c</think><answer>s3</answer>"}]
        ]
        rewards = format_reward_func(completions)

        self.assertEqual(len(rewards), 3)
        self.assertTrue(all(r == 1.0 for r in rewards))

    def test_missing_think(self):
        """Test missing think block gets no reward."""
        completions = [[{"content": "<answer>s1</answer>"}]]
        rewards = format_reward_func(completions)

        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.0)

    def test_missing_answer(self):
        """Test missing answer block gets no reward."""
        completions = [[{"content": "<think>reasoning</think>"}]]
        rewards = format_reward_func(completions)

        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.0)

    def test_empty_think(self):
        """Test empty think block gets no reward."""
        completions = [[{"content": "<think></think><answer>s1</answer>"}]]
        rewards = format_reward_func(completions)

        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.0)

    def test_whitespace_only_think(self):
        """Test whitespace-only think block gets no reward."""
        completions = [[{"content": "<think>   </think><answer>s1</answer>"}]]
        rewards = format_reward_func(completions)

        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.0)

    def test_malformed_xml(self):
        """Test malformed XML gets no reward."""
        completions = [[{"content": "<think>unclosed"}]]
        rewards = format_reward_func(completions)

        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.0)

    def test_plain_text(self):
        """Test plain text without XML gets no reward."""
        completions = [[{"content": "The answer is s1"}]]
        rewards = format_reward_func(completions)

        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.0)

    def test_mixed_valid_invalid(self):
        """Test mix of valid and invalid responses."""
        completions = [
            [{"content": "<think>good</think><answer>s1</answer>"}],
            [{"content": "bad response"}],
            [{"content": "<think>also good</think><answer>s2</answer>"}],
            [{"content": "<answer>missing think</answer>"}]
        ]
        rewards = format_reward_func(completions)

        self.assertEqual(len(rewards), 4)
        self.assertEqual(rewards[0], 1.0)
        self.assertEqual(rewards[1], 0.0)
        self.assertEqual(rewards[2], 1.0)
        self.assertEqual(rewards[3], 0.0)


class TestFinalStateCorrectRewardFunc(unittest.TestCase):
    """Test final state correctness reward function."""

    def test_correct_answer(self):
        """Test correct final state gets full reward."""
        prompts = ["prompt"]
        completions = [[{"content": "<think>reasoning</think><answer>s3</answer>"}]]
        final_state = ["s3"]

        rewards = final_state_correct_reward_func(prompts, completions, final_state)

        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)

    def test_incorrect_answer(self):
        """Test incorrect final state gets no reward."""
        prompts = ["prompt"]
        completions = [[{"content": "<think>reasoning</think><answer>s1</answer>"}]]
        final_state = ["s2"]

        rewards = final_state_correct_reward_func(prompts, completions, final_state)

        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.0)

    def test_multiple_responses(self):
        """Test multiple responses with varying correctness."""
        prompts = ["p1", "p2", "p3"]
        completions = [
            [{"content": "<think>a</think><answer>s1</answer>"}],
            [{"content": "<think>b</think><answer>s2</answer>"}],
            [{"content": "<think>c</think><answer>s3</answer>"}]
        ]
        final_state = ["s1", "s5", "s3"]

        rewards = final_state_correct_reward_func(prompts, completions, final_state)

        self.assertEqual(len(rewards), 3)
        self.assertEqual(rewards[0], 1.0)  # s1 == s1
        self.assertEqual(rewards[1], 0.0)  # s2 != s5
        self.assertEqual(rewards[2], 1.0)  # s3 == s3

    def test_missing_answer(self):
        """Test missing answer gets no reward."""
        prompts = ["prompt"]
        completions = [[{"content": "<think>reasoning</think>"}]]
        final_state = ["s1"]

        rewards = final_state_correct_reward_func(prompts, completions, final_state)

        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.0)

    def test_malformed_response(self):
        """Test malformed response gets no reward."""
        prompts = ["prompt"]
        completions = [[{"content": "bad response"}]]
        final_state = ["s1"]

        rewards = final_state_correct_reward_func(prompts, completions, final_state)

        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.0)

    def test_whitespace_in_answer(self):
        """Test answer with extra whitespace is trimmed correctly."""
        prompts = ["prompt"]
        completions = [[{"content": "<think>reasoning</think><answer>  s2  </answer>"}]]
        final_state = ["s2"]

        rewards = final_state_correct_reward_func(prompts, completions, final_state)

        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)

    def test_case_sensitive_matching(self):
        """Test state matching is case-sensitive."""
        prompts = ["prompt"]
        completions = [[{"content": "<think>reasoning</think><answer>S1</answer>"}]]
        final_state = ["s1"]

        rewards = final_state_correct_reward_func(prompts, completions, final_state)

        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 0.0)

    def test_different_state_formats(self):
        """Test various state naming formats."""
        prompts = ["p1", "p2", "p3"]
        completions = [
            [{"content": "<think>a</think><answer>s10</answer>"}],
            [{"content": "<think>b</think><answer>state_2</answer>"}],
            [{"content": "<think>c</think><answer>q0</answer>"}]
        ]
        final_state = ["s10", "state_2", "q0"]

        rewards = final_state_correct_reward_func(prompts, completions, final_state)

        self.assertEqual(len(rewards), 3)
        self.assertTrue(all(r == 1.0 for r in rewards))


class TestTokenCountRewardFunc(unittest.TestCase):
    """Test token count penalty reward function."""

    def test_within_target(self):
        """Test no penalty for responses within target length."""
        prompts = ["prompt"]
        completion_ids = [[1, 2, 3, 4, 5]]  # 5 tokens

        rewards = token_count_reward_func(
            prompts, completion_ids, target_count=10, decay_rate=256.0
        )

        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)

    def test_exactly_at_target(self):
        """Test no penalty when exactly at target length."""
        prompts = ["prompt"]
        completion_ids = [[1] * 512]  # Exactly 512 tokens

        rewards = token_count_reward_func(
            prompts, completion_ids, target_count=512, decay_rate=256.0
        )

        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)

    def test_exceeds_target(self):
        """Test exponential penalty when exceeding target."""
        prompts = ["prompt"]
        completion_ids = [[1] * 600]  # 600 tokens (88 over target)

        rewards = token_count_reward_func(
            prompts, completion_ids, target_count=512, decay_rate=256.0
        )

        self.assertEqual(len(rewards), 1)
        excess = 600 - 512
        expected_penalty = math.exp(-excess / 256.0)
        self.assertAlmostEqual(rewards[0], expected_penalty, places=6)

    def test_penalty_decay_rate(self):
        """Test that different decay rates produce different penalties."""
        prompts = ["p1", "p2"]
        completion_ids = [[1] * 600, [1] * 600]  # Both 600 tokens

        # Fast decay
        rewards_fast = token_count_reward_func(
            prompts, completion_ids, target_count=512, decay_rate=100.0
        )

        # Slow decay
        rewards_slow = token_count_reward_func(
            prompts, completion_ids, target_count=512, decay_rate=500.0
        )

        # Fast decay should have lower reward (harsher penalty)
        self.assertLess(rewards_fast[0], rewards_slow[0])

    def test_multiple_responses(self):
        """Test penalty for multiple responses with varying lengths."""
        prompts = ["p1", "p2", "p3", "p4"]
        completion_ids = [
            [1] * 100,   # Within target
            [1] * 512,   # Exactly at target
            [1] * 600,   # Over target
            [1] * 1000   # Way over target
        ]

        rewards = token_count_reward_func(
            prompts, completion_ids, target_count=512, decay_rate=256.0
        )

        self.assertEqual(len(rewards), 4)
        self.assertEqual(rewards[0], 1.0)
        self.assertEqual(rewards[1], 1.0)
        self.assertLess(rewards[2], 1.0)
        self.assertLess(rewards[3], rewards[2])  # More excess = lower reward

    def test_exponential_decay_formula(self):
        """Test that penalty follows exponential decay formula."""
        target = 512
        decay = 256.0

        for excess in [50, 100, 200, 400]:
            prompts = ["prompt"]
            completion_ids = [[1] * (target + excess)]

            rewards = token_count_reward_func(
                prompts, completion_ids, target_count=target, decay_rate=decay
            )

            expected = math.exp(-excess / decay)
            self.assertAlmostEqual(rewards[0], expected, places=6)

    def test_zero_tokens(self):
        """Test handling of zero-length completion."""
        prompts = ["prompt"]
        completion_ids = [[]]  # Empty token list

        rewards = token_count_reward_func(
            prompts, completion_ids, target_count=512, decay_rate=256.0
        )

        self.assertEqual(len(rewards), 1)
        self.assertEqual(rewards[0], 1.0)

    def test_custom_target_count(self):
        """Test with custom target count."""
        prompts = ["p1", "p2"]
        completion_ids = [[1] * 100, [1] * 150]

        # Target of 100
        rewards = token_count_reward_func(
            prompts, completion_ids, target_count=100, decay_rate=50.0
        )

        self.assertEqual(rewards[0], 1.0)  # Exactly at target
        expected = math.exp(-50 / 50.0)  # 50 tokens over
        self.assertAlmostEqual(rewards[1], expected, places=6)


class TestIntegration(unittest.TestCase):
    """Integration tests combining multiple reward functions."""

    def test_perfect_response(self):
        """Test a perfect response gets full rewards from all functions."""
        prompts = ["What is the final state?"]
        completions = [[{"content": "<think>Start at s1, apply a, end at s2</think><answer>s2</answer>"}]]
        final_state = ["s2"]
        completion_ids = [[1] * 100]  # Short response

        format_rewards = format_reward_func(completions)
        correctness_rewards = final_state_correct_reward_func(prompts, completions, final_state)
        length_rewards = token_count_reward_func(prompts, completion_ids, target_count=512)

        self.assertEqual(format_rewards[0], 1.0)
        self.assertEqual(correctness_rewards[0], 1.0)
        self.assertEqual(length_rewards[0], 1.0)

    def test_badly_formatted_response(self):
        """Test badly formatted response fails all checks."""
        prompts = ["What is the final state?"]
        completions = [[{"content": "The answer is s2"}]]
        final_state = ["s2"]
        completion_ids = [[1] * 100]

        format_rewards = format_reward_func(completions)
        correctness_rewards = final_state_correct_reward_func(prompts, completions, final_state)
        length_rewards = token_count_reward_func(prompts, completion_ids, target_count=512)

        self.assertEqual(format_rewards[0], 0.0)
        self.assertEqual(correctness_rewards[0], 0.0)
        self.assertEqual(length_rewards[0], 1.0)  # Length is still fine


if __name__ == '__main__':
    unittest.main(verbosity=2)
