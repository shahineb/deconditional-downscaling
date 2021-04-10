#!/bin/bash
#SBATCH --job-name=debug
#SBATCH --time=00:00:30
#SBATCH --partition=ziz-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=14-00:00:00
#SBATCH --mem=4G
#SBATCH --output=/data/ziz/not-backed-up/bouabid/tmp/slurm-%A_%a.o
#SBATCH --error=/data/ziz/not-backed-up/bouabid/tmp/slurm-%A_%a.o

#pyenv activate deconditioning
python experiments/swiss_roll/run_experiment.py --cfg=experiments/swiss_roll/config/exact_cme_process.yaml --o=sandbox/boo_exact_wo --n_epochs=2
python experiments/swiss_roll/run_experiment.py --cfg=experiments/swiss_roll/config/exact_cme_process_indiv_noise.yaml --o=sandbox/boo_exact_w --n_epochs=2
python experiments/swiss_roll/run_experiment.py --cfg=experiments/swiss_roll/config/variational_cme_process.yaml --o=sandbox/boo_var_wo --n_epochs=2
python experiments/swiss_roll/run_experiment.py --cfg=experiments/swiss_roll/config/variational_cme_process_indiv_noise.yaml --o=sandbox/boo_var_w --n_epochs=2
python experiments/swiss_roll/run_experiment.py --cfg=experiments/swiss_roll/config/vbagg.yaml --o=sandbox/boo_vbagg --n_epochs=2

# python experiments/swiss_roll/run_experiment.py --cfg=experiments/swiss_roll/config/exact_cme_process.yaml --o=sandbox/sandbox/exact_wo_noise --plot
# python experiments/swiss_roll/run_experiment.py --cfg=experiments/swiss_roll/config/exact_cme_process_indiv_noise.yaml --o=sandbox/sandbox/exact_w_noise --plot
# python experiments/swiss_roll/run_experiment.py --cfg=experiments/swiss_roll/config/variational_cme_process.yaml --o=sandbox/sandbox/var_wo_noise --plot
# python experiments/swiss_roll/run_experiment.py --cfg=experiments/swiss_roll/config/variational_cme_process_indiv_noise.yaml --o=sandbox/sandbox/var_w_noise --plot --n_epochs=300
# python experiments/swiss_roll/run_experiment.py --cfg=experiments/swiss_roll/config/vbagg.yaml --o=sandbox/sandbox/vbagg --plot
echo "Job Completed"
