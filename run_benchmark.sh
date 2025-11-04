#!/bin/bash
#SBATCH -J dvi-h5
#SBATCH --output=logs/clews_benchmark_%j.out
#SBATCH --error=logs/clews_benchmark_%j.err
#SBATCH -N 1
#SBATCH -t 48:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16

module purge
module load CUDA/12.1.1          # or whatever version appears in `module spider`
module load Anaconda3/2024.02-1


source $(conda info --base)/etc/profile.d/conda.sh
conda activate clewsTesting

export OMP_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE

srun python -u train_h5.py jobname=dvi-h5 conf=config/dvi-h5.yaml fabric.nnodes=1 fabric.ngpus=2
