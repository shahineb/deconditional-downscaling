#!/bin/bash
#SBATCH --job-name=swiss_roll_sweep
#SBATCH --time=00:00:30
#SBATCH --partition=ziz-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=14-00:00:00
#SBATCH --mem=4G
#SBATCH --output=/data/ziz/not-backed-up/bouabid/tmp/slurm-%A_%a.o
#SBATCH --error=/data/ziz/not-backed-up/bouabid/tmp/slurm-%A_%a.o

#pyenv activate deconditioning
#python experiments/swiss_roll/run_experiment.py --cfg=experiments/swiss_roll/config/exact_cme_process.yaml --o=sandbox/boo --n_epochs=1
#python experiments/swiss_roll/run_experiment.py --cfg=experiments/swiss_roll/config/variational_cme_process.yaml --o=sandbox/boo_1 --n_epochs=1
python experiments/swiss_roll/run_experiment.py --cfg=experiments/swiss_roll/config/vbagg.yaml --o=sandbox/boo_2 --n_epochs=1
echo "Job Completed"

