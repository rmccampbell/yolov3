#!/usr/bin/env bash
#SBATCH -o test_yolo_%j.out
#SBATCH -e test_yolo_%j.err
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 1:00:00
#SBATCH -A class-cs6501-004-sp19

module purge
module load singularity
# module load tensorflow

# Assuming that the container has been copied to the user's /scratch directory
containerdir=/scratch/$USER

singularity exec --nv \
    $containerdir/tensorflow-1.12.0-py36.simg \
    python -u test.py --data data/svhn.data --cfg cfg/yolov3-svhn.cfg --weights weights/best.pt
