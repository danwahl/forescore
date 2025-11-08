import datasets
import logging
import os
import re
import sys
import torch
import transformers

from dataclasses import dataclass, field
from datasets import Dataset
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
    final_state_correct_reward_func,
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
    dataset_name: str = field(default="./data/fsm_s3-10_l5-20_n10000")
    dataset_subset: str = field(default=None)
    name: str = field(
        default="forscore",
    )
    system_prompt: str = field(
        default="""You are an expert at simulating finite state machines (FSMs). When given an FSM definition in DOT notation and a sequence of inputs, you need to determine the final state after processing the entire sequence.

Think through the problem step-by-step:
1. Identify the initial state (marked by Start arrow)
2. Process each input symbol one at a time
3. Follow the transitions according to the FSM's edge labels
4. Track your current state after each transition

Here's an example showing the input format and required response:

Input:
digraph FSM {
  rankdir=LR;
  node [shape=circle];

  Start [shape=point];
  Start -> S0;

  S0 -> S1 [label="a"];
  S0 -> S2 [label="b"];
  S1 -> S0 [label="a"];
  S1 -> S3 [label="b"];
  S2 -> S3 [label="a"];
  S2 -> S0 [label="b"];
  S3 -> S2 [label="a"];
  S3 -> S1 [label="b"];
}

Starting from the initial state, process this sequence of inputs:
b, a, b, a, b

What is the final state?

Response:
<think>
Looking at the FSM:
- Initial state: Start → S0
- Transitions define how each state responds to inputs 'a' and 'b'

Processing the sequence: b, a, b, a, b

Step 1: Start at S0
Step 2: Input 'b' → S0 -> S2 [label="b"] → now at S2
Step 3: Input 'a' → S2 -> S3 [label="a"] → now at S3
Step 4: Input 'b' → S3 -> S1 [label="b"] → now at S1
Step 5: Input 'a' → S1 -> S0 [label="a"] → now at S0
Step 6: Input 'b' → S0 -> S2 [label="b"] → now at S2

Final state after processing all inputs: S2
</think>
<answer>S2</answer>

Your response MUST follow this XML format:
<think>
[Your step-by-step reasoning here - trace through each state transition]
</think>
<answer>
[Final state name, e.g., S0, S1, S2, etc.]
</answer>

Be thorough in your thinking process, showing each state transition clearly.
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
    """
    Prepare FSM traversal examples for GRPO training.

    Expected example fields from dataset:
    - problem: Full FSM problem statement (DOT + sequence + question)
    - final_state: Ground truth final state
    - trace: List of states visited (for debugging)
    - sequence: Input sequence as list
    """
    problem = example["problem"]
    final_state = example["final_state"]

    return {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": problem},
        ],
        "final_state": final_state,
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
    if not os.path.exists(pt_args.dataset_name):
        logger.error(f"Dataset not found at: {pt_args.dataset_name}")
        logger.error("Please generate the dataset first using generate.py")
        sys.exit(1)

    logger.info(f"Loading dataset from local path: {pt_args.dataset_name}")
    train_dataset = Dataset.load_from_disk(pt_args.dataset_name)

    # Preprocess dataset
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
            final_state_correct_reward_func,
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
