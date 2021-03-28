#!/bin/bash
#SBATCH --job-name=cloud-sweep                               
#SBATCH --time=00:00:30                                
#SBATCH --partition=ziz-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=14-00:00:00
#SBATCH --mem=2G
#SBATCH --output=/data/ziz/not-backed-up/bouabid/tmp/slurm-%A_%a.o         
#SBATCH --error=/data/ziz/not-backed-up/bouabid/tmp/slurm-%A_%a.o         

pyenv activate deconditioning
python /data/ziz/not-backed-up/bouabid/repos/ContBagGP/experiments/downscaling/run_experiment.py --cfg=/data/ziz/not-backed-up/bouabid/repos/ContBagGP/experiments/downscaling/config/variational_cme_process.yaml --o=/data/ziz/not-backed-up/bouabid/tmp
echo "Job Completed"
