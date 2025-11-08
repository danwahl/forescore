# forscore

Can LLMs think while you recite the Gettysburg Address?

## Setup

1. `srun -p general --constraint=a100 --gres=gpu:2 --cpus-per-task=32 --pty --mem 64G -t 270:00 /bin/bash`
2. `conda create -n forscore python=3.12 -y`
3. `conda activate forscore`
4. `pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121`
5. `export PATH=/usr/local/cuda-12.1/bin:$PATH`
6. `MAX_JOBS=8 pip install flash-attn==2.7.4.post1 --no-build-isolation'

## Train

`bash slurm/launch.sh slurm/train.slurm`

### Manual instructions

1. Configure Wandb:
    ```
    export WANDB_API_KEY=...
    export WANDB_CACHE_DIR=/net/scratch/$USER/wandb
    ```
2. `accelerate launch --config_file configs/accelerate.yaml train.py --config configs/forscore.yaml --base-dir /net/scratch/$USER/post-training/forscore --log-level info`
