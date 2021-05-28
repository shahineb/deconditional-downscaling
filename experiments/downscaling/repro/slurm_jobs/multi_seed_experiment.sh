#!/bin/bash
#SBATCH --job-name=seedroll
#SBATCH --partition=ziz-gpu
#SBATCH --gres=gpu:2
#SBATCH --nodelist=zizgpu04.cpu.stats.ox.ac.uk
#SBATCH --time=14-00:00:00
#SBATCH --mem=14G
#SBATCH --output=/data/ziz/not-backed-up/bouabid/tmp/slurm-%A_%a.o
#SBATCH --error=/data/ziz/not-backed-up/bouabid/tmp/slurm-%A_%a.o

pyenv activate deconditioning
bash /data/ziz/not-backed-up/bouabid/repos/ContBagGP/experiments/downscaling/repro/run_multi_seed_experiment.sh
echo "Job Completed"
