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

# # python create_pointcloud.py --mesh data/armadillo/armadillo.obj
# python train_igr.py \
#     --input data/armadillo/armadillo_100000.npz


# python create_pointcloud.py --mesh data/bunny/bunny.obj
python train_igr.py \
    --input data/bunny/bunny_watertight_100000.npz


# # python create_pointcloud.py --mesh data/cow/cow.obj
# python train_igr.py \
#     --input data/cow/cow_100000.npz


# # python create_pointcloud.py --mesh data/dragon/dragon.obj
# python train_igr.py \
#     --input data/dragon/dragon_100000.npz


# # python create_pointcloud.py --mesh data/teapot/teapot.obj
# python train_igr.py \
#     --input data/teapot/teapot_100000.npz


echo "======================================================"
echo "Job finished"
echo "======================================================"
