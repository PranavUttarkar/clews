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
module load cuda/12.0          # or whatever version appears in `module spider`
module load anaconda3/2023.09  # update version if different

conda activate clewsTesting

export OMP_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE

srun python -u train_h5.py jobname=dvi-h5 conf=config/dvi-h5.yaml fabric.nnodes=1 fabric.ngpus=2
