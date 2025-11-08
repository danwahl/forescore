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
            "centipawn": None,
            "turn": None,
            "best_move": None,
            "piece_count": None,
            "white_king": None,
        }

        if answer_elem is not None:
            for field in [
                "centipawn",
                "turn",
                "best_move",
                "piece_count",
                "white_king",
            ]:
                elem = answer_elem.find(field)
                result[field] = (
                    elem.text.strip() if elem is not None and elem.text else None
                )

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
                # Check if we have think and all required answer fields
                has_think = parsed["think"] is not None
                has_all_fields = all(
                    parsed[field] is not None
                    for field in ["centipawn", "turn", "best_move"]
                )
                results.append(1.0 if has_think and has_all_fields else 0.0)
            except:
                results.append(0.0)

    return results


def centipawn_int_reward_func(completions, **kwargs) -> list[float]:
    """Reward for centipawn being a legal integer."""
    responses = [completion[0]["content"] for completion in completions]
    results = []

    for response in responses:
        parsed = parse_full_response(response)
        if parsed is None or parsed["centipawn"] is None:
            results.append(0.0)
        else:
            try:
                int(parsed["centipawn"])
                results.append(1.0)
            except:
                results.append(0.0)

    return results


def centipawn_accuracy_reward_func(prompts, completions, cp, **kwargs) -> list[float]:
    """Reward based on centipawn accuracy using exponential decay."""
    responses = [completion[0]["content"] for completion in completions]
    results = []

    for response, target_cp in zip(responses, cp):
        parsed = parse_full_response(response)
        if parsed is None or parsed["centipawn"] is None:
            results.append(0.0)
        else:
            try:
                predicted_cp = int(parsed["centipawn"])
                # Use exponential decay: e^(-error/100)
                # This gives smoother rewards that decay more gracefully
                error = abs(target_cp - predicted_cp)
                reward = math.exp(-error / 100.0)
                results.append(reward)
            except:
                results.append(0.0)

    return results


def turn_reward_func(prompts, completions, turn, **kwargs) -> list[float]:
    """Reward for correct turn prediction."""
    responses = [completion[0]["content"] for completion in completions]
    results = []

    for response, target_turn in zip(responses, turn):
        parsed = parse_full_response(response)
        if parsed is None or parsed["turn"] is None:
            results.append(0.0)
        else:
            try:
                results.append(1.0 if parsed["turn"].lower() == target_turn else 0.0)
            except Exception:
                results.append(0.0)

    return results


def best_move_legal_reward_func(
    prompts, completions, legal_moves, **kwargs
) -> list[float]:
    """Reward for predicting a legal move."""
    responses = [completion[0]["content"] for completion in completions]
    results = []

    for response, legal_move_list in zip(responses, legal_moves):
        parsed = parse_full_response(response)
        if parsed is None or parsed["best_move"] is None:
            results.append(0.0)
        else:
            try:
                results.append(1.0 if parsed["best_move"] in legal_move_list else 0.0)
            except Exception:
                results.append(0.0)

    return results


def best_move_correct_reward_func(
    prompts, completions, best_move, **kwargs
) -> list[float]:
    """Reward for predicting the correct best move."""
    responses = [completion[0]["content"] for completion in completions]
    results = []

    for response, target_move in zip(responses, best_move):
        parsed = parse_full_response(response)
        if parsed is None or parsed["best_move"] is None or target_move is None:
            results.append(0.0)
        else:
            try:
                results.append(1.0 if parsed["best_move"] == target_move else 0.0)
            except Exception:
                results.append(0.0)

    return results


def piece_count_reward_func(prompts, completions, piece_count, **kwargs) -> list[float]:
    """Reward for correctly counting the total number of pieces on the board."""
    responses = [completion[0]["content"] for completion in completions]
    results = []

    for response, actual_count in zip(responses, piece_count):
        parsed = parse_full_response(response)
        if parsed is None or parsed["piece_count"] is None:
            results.append(0.0)
        else:
            try:
                # Compare with model's count
                predicted_count = int(parsed["piece_count"])

                # Give full reward for exact match, partial for being close
                if predicted_count == actual_count:
                    results.append(1.0)
                elif abs(predicted_count - actual_count) == 1:
                    results.append(0.5)  # Partial credit for off-by-one
                else:
                    results.append(0.0)
            except Exception:
                results.append(0.0)

    return results


def white_king_location_reward_func(
    prompts, completions, white_king, **kwargs
) -> list[float]:
    """Reward for correctly identifying the location of the white king."""
    responses = [completion[0]["content"] for completion in completions]
    results = []

    for response, target_location in zip(responses, white_king):
        parsed = parse_full_response(response)
        if parsed is None or parsed["white_king"] is None:
            results.append(0.0)
        else:
            try:
                results.append(1.0 if parsed["white_king"] == target_location else 0.0)
            except Exception:
                results.append(0.0)

    return results


def token_count_reward_func(
    prompts, completion_ids, target_count=1024, decay_rate=512.0, **kwargs
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
