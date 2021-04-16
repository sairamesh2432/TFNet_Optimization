#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks-per-node=1 
#SBATCH --mem=5gb 
#SBATCH --time=15:00:00
#SBATCH --output=your_job_name.out  


# pay close attention to memory as you increase the gpu count



module purge

module load CUDA
module load GCC/6.4.0-2.28  OpenMPI/2.1.2-CUDA
module load GCC/8.3.0  CUDA/10.1.243
module load NCCL/2.6.4
# module load Python/3.6.4
module load Anaconda/3

conda activate base

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1


srun python lightning_TF_train.py -region ./region.txt -LR 0.001 -output ./test_${SLURM_JOB_ID}

scontrol show job $SLURM_JOB_ID
