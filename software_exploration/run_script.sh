#!/bin/bash
#SBATCH --gres=gpu:v100:1 
#SBATCH --mem=5gb 
#SBATCH --time=15:00:00


module purge
module load GCC/6.4.0-2.28 OpenMPI/2.1.2
module load CUDA/9.0.176 cuDNN/7.0.2-CUDA-9.0.176
module load Python/3.6.4
module load Anaconda/3

conda activate ./envs 
python lightning_TF_train.py -region ./region.txt -LR 0.001 -output ./third_test_${SLURM_JOB_ID} 

scontrol show job $SLURM_JOB_ID
