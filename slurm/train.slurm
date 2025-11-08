#!/bin/bash

#SBATCH --job-name=forscore_train
#SBATCH --partition=general
#SBATCH --account=general_group
#SBATCH --output=output-%x.%j
#SBATCH --error=error-%x.%j
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=a100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=i001-ds

# Set environment

ulimit -s unlimited

CONDA_ENV="forscore"
source /net/scratch/$USER/miniconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV
echo "START TIME: $(date)"
echo "PYTHON ENV: $(which python)"

export CONFIG="forscore"
export HF_TOKEN="" # Set this
export WANDB_API_KEY=""  # Set this
export WANDB_CACHE_DIR="/net/scratch/$USER/wandb"
export BASE_DIR="/net/scratch/$USER/post-training/forscore"

wandb login $WANDB_API_KEY

export CMD="accelerate launch --config_file configs/accelerate.yaml train.py --config configs/$CONFIG.yaml --base-dir $BASE_DIR --log-level info --overwrite_output_dir false"
srun $CMD

echo ""
echo "END TIME: $(date)"
echo "DONE"
