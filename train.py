import chess
import datasets
import logging
import os
import re
import sys
import torch
import transformers

from dataclasses import dataclass, field
from datasets import load_dataset, Dataset
from datetime import datetime
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Gemma3ForConditionalGeneration,
)
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOConfig, GRPOTrainer, ModelConfig, TrlParser

from rewards import (
    PAD_TOKEN,
    format_reward_func,
    centipawn_int_reward_func,
    centipawn_accuracy_reward_func,
    turn_reward_func,
    best_move_legal_reward_func,
    best_move_correct_reward_func,
    piece_count_reward_func,
    white_king_location_reward_func,
    token_count_reward_func,
)


@dataclass
class PTConfig:
    base_dir: str = (
        field(
            default=".",
        ),
    )
    chat_template: str = field(
        default="""{%- if messages and messages[0]['role'] == 'system' -%}
    {%- set conversation = messages -%}
{%- else -%}
    {%- set conversation = [{'role': 'system', 'content': 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'}] + (messages if messages else []) -%}
{%- endif -%}
{%- for message in conversation -%}
    {%- if message.role == 'system' -%}
        {{- '<|im_start|>system\n' + message.content + '<|im_end|>\n' -}}
    {%- elif message.role == 'user' -%}
        {{- '<|im_start|>user\n' + message.content + '<|im_end|>\n' -}}
    {%- elif message.role == 'assistant' -%}
        {{- '<|im_start|>assistant\n' + (message.content | default('', true)) + '<|im_end|>\n' -}}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{- '<|im_start|>assistant\n' -}}
{%- endif -%}"""
    )
    dataset_name: str = field(default="Lichess/chess-position-evaluations")
    dataset_subset: str = field(default=None)
    name: str = field(
        default="forscore",
    )
    system_prompt: str = field(
        default="""You are a helpful chess assistant. When given a FEN, think carefully about the position, then provide the following information:

1. Whose turn it is to move (White or Black)
2. The total number of pieces on the board
3. The location of the White king (e.g. e1, g1)
4. The best move in the position (in UCI notation, e.g, e2e4, g8f6)
5. A centipawn evaluation (100 cp = 1 pawn advantage, positive for White advantage, negative for Black advantage)
6. A one-sentence analysis of the position, to be displayed to the user

Here's an example response showing the required XML format:
<think>
The user provided the FEN: r1bqk2r/ppp1bppp/2n2n2/3pP3/3P4/2N1BN2/PPP2PPP/R2QKB1R w Kq d6

Parsing FEN ranks 8→1:
8: r1bqk2r → ra8, bc8, qd8, ke8, rh8
7: ppp1bppp → pawns a7,b7,c7,f7,g7,h7 + be7
6: 2n2n2 → nc6, nf6
5: 3pP3 → Black pawn d5, White pawn e5
4: 3P4 → White pawn d4
3: 2N1BN2 → Nc3, Be3, Nf3
2: PPP2PPP → pawns a2,b2,c2,f2,g2,h2
1: R2QKB1R → Ra1, Qd1, Ke1, Bf1, Rh1

Pieces: Ra1,Nc3,Be3,Qd1,Ke1,Bf1,Nf3,Rh1 + 6 pawns (White); ra8,bc8,qd8,ke8,be7,nc6,nf6,rh8 + 6 pawns (Black) = 28 total.

Turn: w = White. White king: e1.
Castling: Kq = White can castle kingside, Black can castle queenside only.
En passant: d6 = Black just played d7-d5.

Move options: e5f6 (capture knight), e5d6 (en passant capture), e1g1 (kingside castle), or piece moves like c3b5. The pawn capture e5f6 wins a full knight (worth ~3 pawns) versus just winning a pawn with e5d6. This makes e5f6 clearly the best move.
</think>
<answer>
<turn>White</turn>
<piece_count>28</piece_count>
<white_king>e1</white_king>
<best_move>e5f6</best_move>
<centipawn>+441</centipawn>
<analysis>
White should capture the knight with e5f6, winning material in this developed middlegame.
</analysis>
</answer>
"""
    )


# Your entire response MUST be in the exact XML format below:
# <think>
# [...thinking process here...]
# </think>
# <answer>
# <turn>...</turn>
# <piece_count>...</piece_count>
# <white_king>...</white_king>
# <best_move>...</best_move>
# <centipawn>...</centipawn>
# <analysis>
# ...
# </analysis>
# </answer>

# In your thinking process:
# - Look at the board and assess the material balance, identifying which side has more pieces and their types
# - Analyze the position for tactical and strategic elements, such as piece activity, king safety, pawn structure, and control of key squares
# - Consider candidate moves and explain why certain moves are better than others
# - Evaluate the current position in centipawns (100 cp = 1 pawn advantage, positive favors white, negative favors black)

# Then provide:
# 1. Which side is to move (white or black)
# 2. The total number of pieces on the board
# 3. The location of the white king (in UCI notation, e.g., e1, g1)
# 4. The best move in the position (in UCI notation, e.g., e2e4, g8f6)
# 5. Your centipawn evaluation


def make_conv_for_grpo(example, system_prompt):
    fen = example["fen"]
    board = chess.Board(fen)

    # # Create board visualization
    # board_str = str(board)
    # # board_str = "▫" + board_str + "▫"
    # # board_str = re.sub(r" ", "▫", board_str)
    # # board_str = re.sub(r"\n", f"▫\n▫", board_str)

    # # Add row labels to board string
    # rows = board_str.split('\n')
    # for i, row in enumerate(rows):
    #     # Chess board labels go from 8 to 1 (top to bottom)
    #     row_label = str(8 - i)
    #     rows[i] = row_label + ' ' + row
    # board_str = '\n'.join(rows)

    # # Add column labels at the top
    # col_labels = '  a b c d e f g h'
    # board_str = col_labels + '\n' + board_str

    # can_castle_kingside = board.has_kingside_castling_rights(board.turn)
    # can_castle_queenside = board.has_queenside_castling_rights(board.turn)
    # ep_square = board.ep_square

    # # Add additional information to the board string
    # board_str += f"\nTo move: {'white' if board.turn else 'black'}"
    # board_str += f"\nCan castle kingside: {'yes' if can_castle_kingside else 'no'}"
    # board_str += f"\nCan castle queenside: {'yes' if can_castle_queenside else 'no'}"
    # board_str += f"\nEn passant square: {chess.square_name(ep_square) if ep_square is not None else 'N/A'}"

    # Extract preprocessing fields using chess board
    legal_moves = [move.uci() for move in board.legal_moves]
    turn = "white" if board.turn else "black"  # board.turn is True for white
    piece_count = len(board.piece_map())  # Pre-calculate piece count
    white_king = chess.SQUARE_NAMES[board.king(chess.WHITE)]

    best_move = None
    if example.get("line"):
        moves = example["line"].split()
        best_move = moves[0] if moves else None

    return {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": fen},
        ],
        "legal_moves": legal_moves,
        "turn": turn,
        "best_move": best_move,
        "piece_count": piece_count,
        "white_king": white_king,
    }


def main():
    parser = TrlParser((PTConfig, GRPOConfig, ModelConfig))
    pt_args, training_args, model_args = parser.parse_args_and_config()

    # os.environ["RANK"] = "0"
    # os.environ["WORLD_SIZE"] = "1"
    # os.environ["LOCAL_RANK"] = "0"
    # os.environ["MASTER_ADDR"] = "localhost"
    # os.environ["MASTER_PORT"] = "12356"
    os.environ["WANDB_PROJECT"] = "forscore"

    # Set up logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)

    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Post training parameters {pt_args}")
    logger.info(f"Training parameters {training_args}")

    # Set up output paths
    current_time = datetime.now()
    formatted_datetime = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    training_args.logging_dir = f"{pt_args.base_dir}/{pt_args.name}/logs"
    training_args.output_dir = f"{pt_args.base_dir}/{pt_args.name}/checkpoints"
    training_args.run_name = f"{pt_args.name}_{formatted_datetime}"

    # Load and preprocess dataset (tokenization is handled by GRPO Trainer)
    streaming_dataset = (
        load_dataset(
            pt_args.dataset_name, pt_args.dataset_subset, split="train", streaming=True
        )
        .shuffle(seed=training_args.seed)
        .take(10000)
    )
    train_dataset = Dataset.from_list(list(streaming_dataset))
    train_dataset = train_dataset.map(
        make_conv_for_grpo, fn_kwargs={"system_prompt": pt_args.system_prompt}
    )
    train_dataset.save_to_disk(f"{pt_args.base_dir}/{pt_args.name}/data")

    # Initialize the model
    model = AutoModelForCausalLM.from_pretrained(
        # model = Gemma3ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        # attn_implementation=model_args.attn_implementation,
        # use_cache=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    # tokenizer.pad_token = PAD_TOKEN
    tokenizer.chat_template = pt_args.chat_template

    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        target_modules=model_args.lora_target_modules,
        bias="none",
    )

    # Initialize the GRPO trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            format_reward_func,
            centipawn_int_reward_func,
            centipawn_accuracy_reward_func,
            turn_reward_func,
            best_move_legal_reward_func,
            best_move_correct_reward_func,
            piece_count_reward_func,
            white_king_location_reward_func,
            token_count_reward_func,
        ],
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
    )

    # Training and Evaluation
    logger.info(f"\nStarting training for {training_args.num_train_epochs} epochs.")

    # Check for last checkpoint
    ckpt = None
    if training_args.resume_from_checkpoint is not None:
        ckpt = training_args.resume_from_checkpoint
    elif os.path.isdir(training_args.output_dir):
        ckpt = get_last_checkpoint(training_args.output_dir)
        if ckpt:
            logger.info(f"\nCheckpoint detected, resuming training at {ckpt=}.")
        else:
            logger.info("\nNo checkpoint detected, starting training from scratch.")

    try:
        train_result = trainer.train(resume_from_checkpoint=ckpt)
        train_metrics = train_result.metrics
        trainer.log_metrics("train", train_metrics)
        trainer.save_metrics("train", train_metrics)
        trainer.save_state()
    finally:
        del trainer
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
