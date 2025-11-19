#!/bin/bash
#
#SBATCH --job-name=aris_train
#SBATCH --account=class
#SBATCH --partition=class
#SBATCH --output=slurm_logs/slurm-%j.out
#SBATCH --error=slurm_logs/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-00:30:00
#SBATCH --gres=gpu:1

mkdir -p slurm_logs

echo "======================================================"
echo "Starting job      : $SLURM_JOB_ID"
echo "Running on node   : $SLURMD_NODENAME"
echo "Assigned GPU      : $CUDA_VISIBLE_DEVICES"
echo "======================================================"

conda init
# source $(conda info --base)/etc/profile.d/conda.sh
conda activate aris-3

python train.py \
    --config-name train_render \
    scene=cbox_train_diffuse \
    gt=./scripts/cbox-diffuse.exr \
    device=cuda

echo "======================================================"
echo "Job finished"
echo "======================================================"
