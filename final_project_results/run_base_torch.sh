#!/bin/bash
#SBATCH -N 4
#SBATCH --mem=5gb 
#SBATCH --time=15:00:00
#SBATCH --output=serial_test

TRAIN_SCRIPT=$1
OUTPUT=$2

module purge
module load GCC/6.4.0-2.28 OpenMPI/2.1.2
module load CUDA/9.0.176 cuDNN/7.0.2-CUDA-9.0.176
module load Python/3.6.4
module load Anaconda/3

#conda init bash
conda activate base
srun python TF_train.py --gpus 4 -region ./region.txt -LR 0.001 -output ./seventh_test

scontrol show job $SLURM_JOB_ID
