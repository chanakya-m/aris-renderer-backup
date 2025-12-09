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

cd /fs/classhomes/cmaddine/aris-renderer-backup/igr

python train_igr.py \
    --input data/bunny/bunny_watertight_100000.npz

echo "======================================================"
echo "Job finished"
echo "======================================================"
