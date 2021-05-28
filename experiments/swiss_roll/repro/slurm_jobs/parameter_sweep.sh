#!/bin/bash
#SBATCH --job-name=swiss_roll_sweep
#SBATCH --partition=ziz-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=14-00:00:00
#SBATCH --mem=2G
#SBATCH --output=/data/ziz/not-backed-up/bouabid/tmp/slurm-%A_%a.o
#SBATCH --error=/data/ziz/not-backed-up/bouabid/tmp/slurm-%A_%a.o

pyenv activate deconditioning
bash /data/ziz/not-backed-up/bouabid/repos/ContBagGP/experiments/swiss_roll/repro/run_parameter_sweep.sh
echo "Job Completed"
