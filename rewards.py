import xml.etree.ElementTree as ET
import math

PAD_TOKEN = "<|fim_pad|>"


def parse_full_response(text: str) -> dict:
    """Parse the full XML response including think and answer blocks."""
    try:
        # Remove any leading/trailing whitespace and padding tokens
        text = text.replace(PAD_TOKEN, "").strip()

        # Wrap in root element since XML needs single root
        wrapped = f"<root>{text}</root>"
        root = ET.fromstring(wrapped)

        think_elem = root.find("think")
        answer_elem = root.find("answer")

        result = {
            "think": think_elem.text if think_elem is not None else None,
            "final_state": None,
        }

        if answer_elem is not None:
            # For FSM task, we just need the final state
            # The answer could be just text or wrapped in a tag
            if answer_elem.text and answer_elem.text.strip():
                result["final_state"] = answer_elem.text.strip()

        return result
    except:
        return None


def format_reward_func(completions, **kwargs) -> list[float]:
    """Reward for proper XML format with think and answer blocks."""
    responses = [completion[0]["content"] for completion in completions]
    results = []

    for response in responses:
        parsed = parse_full_response(response)
        if parsed is None:
            results.append(0.0)
        else:
            try:
                # Check if we have both think and answer
                has_think = parsed["think"] is not None and len(parsed["think"].strip()) > 0
                has_answer = parsed["final_state"] is not None
                results.append(1.0 if has_think and has_answer else 0.0)
            except:
                results.append(0.0)

    return results


def final_state_correct_reward_func(
    prompts, completions, final_state, **kwargs
) -> list[float]:
    """Reward for predicting the correct final state."""
    responses = [completion[0]["content"] for completion in completions]
    results = []

    for response, target_state in zip(responses, final_state):
        parsed = parse_full_response(response)
        if parsed is None or parsed["final_state"] is None:
            results.append(0.0)
        else:
            try:
                # Exact match for final state
                predicted_state = parsed["final_state"].strip()
                results.append(1.0 if predicted_state == target_state else 0.0)
            except Exception:
                results.append(0.0)

    return results


def token_count_reward_func(
    prompts, completion_ids, target_count=512, decay_rate=256.0, **kwargs
) -> list[float]:
    """Penalty for responses that exceed target length using exponential decay."""
    results = []

    for tokens in completion_ids:
        count = len(tokens)

        # No penalty if within target count
        if count <= target_count:
            results.append(1.0)
        else:
            # Exponential decay penalty for exceeding target
            excess = count - target_count
            penalty = math.exp(-excess / decay_rate)
            results.append(penalty)

    return results
