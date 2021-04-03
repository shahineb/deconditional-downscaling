#!/bin/bash
#SBATCH --job-name=coarseness
#SBATCH --partition=ziz-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=14-00:00:00
#SBATCH --mem=3G
#SBATCH --output=/data/ziz/not-backed-up/bouabid/tmp/slurm-%A_%a.o
#SBATCH --error=/data/ziz/not-backed-up/bouabid/tmp/slurm-%A_%a.o

pyenv activate deconditioning
bash /data/ziz/not-backed-up/bouabid/repos/ContBagGP/experiments/downscaling/repro/run_dummy_coarseness_experiment.sh
echo "Job Completed"
