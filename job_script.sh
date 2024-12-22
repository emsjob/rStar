#!/bin/sh

#BSUB -q gpua100
#BSUB -J llm_job
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 22:00
#BSUB -R "rusage[mem=12GB]"
#BSUB -R "select[gpu40gb]"
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o run_output_%J.out
#BSUB -e err_output_%J.err

export HUGGINGFACE_HUB_TOKEN=<huggingface token>
export TMPDIR="tmp"
mkdir -p tmp

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

module load python3/3.10.13
module load cuda/11.6
module load cudnn/v8.8.0-prod-cuda-12.X

cd /path/to/dir/rStar/rStar
source .venv/bin/activate

python /path/to/dir/rStar/rStar/run_src/do_generate.py \
    --dataset_name GSM8K \
    --test_json_filename test_sampled \
    --model_ckpt mistralai/Mistral-7B-v0.1 \
    --note default \
    --num_rollouts 32 \
    --num_subquestions 5 \
    --num_a1_steps 5 \
    --disable_a5
