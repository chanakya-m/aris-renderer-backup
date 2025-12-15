#!/bin/bash
#
#SBATCH --job-name=rec
#SBATCH --account=class
#SBATCH --partition=class
#SBATCH --output=slurm_logs/slurm-%j.out
#SBATCH --error=slurm_logs/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0-00:30:00
#SBATCH --gres=gpu:1

module load cuda/12.1.1

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

# python reconstruct_igr.py \
#     --checkpoint checkpoints/igr_final_armadillo_100000_LRL.pth \
#     --output data/armadillo/armadillo_100000_reconstructed.ply \
#     --resolution 256

python reconstruct_igr.py \
    --checkpoint checkpoints/igr_exp_bunny_watertight_100000.pth \
    --output data/bunny/bunny_watertight_100000_reconstructed_LRL.ply \
    --resolution 256

# python reconstruct_igr.py \
#     --checkpoint checkpoints/igr_final_cow_100000_LRL.pth \
#     --output data/cow/cow_100000_reconstructed.ply \
#     --resolution 256

# python reconstruct_igr.py \
#     --checkpoint checkpoints/igr_final_dragon_100000_LRL.pth \
#     --output data/dragon/dragon_100000_reconstructed.ply \
#     --resolution 256

# python reconstruct_igr.py \
#     --checkpoint checkpoints/igr_final_teapot_100000_LRL.pth \
#     --output data/teapot/teapot_100000_reconstructed.ply \
#     --resolution 256

echo "======================================================"
echo "Job finished"
echo "======================================================"
