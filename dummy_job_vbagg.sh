#!/bin/bash
#SBATCH --job-name=fdowncloud
#SBATCH --partition=ziz-gpu
#SBATCH --gres=gpu:2
#NOSBATCH --nodelist=zizgpu03.cpu.stats.ox.ac.uk
#SBATCH --time=14-00:00:00
#SBATCH --mem=12G
#SBATCH --output=/data/ziz/not-backed-up/bouabid/tmp/slurm-%A_%a.o
#SBATCH --error=/data/ziz/not-backed-up/bouabid/tmp/slurm-%A_%a.o


# python experiments/downscaling/run_experiment.py --cfg=experiments/downscaling/config/variational_cme_process.yaml --o=sandbox/boo/boo_var_wo --n_epochs=2 --plot
# python experiments/downscaling/run_experiment.py --cfg=experiments/downscaling/config/variational_cme_process_indiv_noise.yaml --o=sandbox/boo/boo_var_w --n_epochs=2 --plot
# python experiments/downscaling/run_experiment.py --cfg=experiments/downscaling/config/vbagg.yaml --o=sandbox/boo/boo_vbagg --n_epochs=2 --plot
# python experiments/downscaling/run_experiment.py --cfg=experiments/downscaling/config/krigging.yaml --o=sandbox/boo/boo_krigging --n_epochs=2 --plot


# python experiments/downscaling/run_experiment.py --cfg=experiments/downscaling/config/variational_cme_process.yaml --o=sandbox/sandbox/boo_var_wo --plot
# python experiments/downscaling/run_experiment.py --cfg=experiments/downscaling/config/variational_cme_process_indiv_noise.yaml --o=sandbox/sandbox/boo_var_w --plot
python experiments/downscaling/run_experiment.py --cfg=experiments/downscaling/config/vbagg.yaml --o=sandbox/sandbox/boo_vbagg --plot
# python experiments/downscaling/run_experiment.py --cfg=experiments/downscaling/config/krigging.yaml --o=sandbox/sandbox/boo_krigging --plot

#pyenv activate deconditioning
# python experiments/swiss_roll/run_experiment.py --cfg=experiments/swiss_roll/config/exact_cme_process.yaml --o=sandbox/boo/boo_exact_wo --n_epochs=2
# python experiments/swiss_roll/run_experiment.py --cfg=experiments/swiss_roll/config/exact_cme_process_indiv_noise.yaml --o=sandbox/boo/boo_exact_w --n_epochs=2
# python experiments/swiss_roll/run_experiment.py --cfg=experiments/swiss_roll/config/variational_cme_process.yaml --o=sandbox/boo/boo_var_wo --n_epochs=2
# python experiments/swiss_roll/run_experiment.py --cfg=experiments/swiss_roll/config/variational_cme_process_indiv_noise.yaml --o=sandbox/boo/boo_var_w --n_epochs=2
# python experiments/swiss_roll/run_experiment.py --cfg=experiments/swiss_roll/config/vbagg.yaml --o=sandbox/boo/boo_vbagg --n_epochs=2
# python experiments/swiss_roll/run_experiment.py --cfg=experiments/swiss_roll/config/gp_regression.yaml --o=sandbox/boo/boo_gp_regression --n_epochs=2

# python experiments/swiss_roll/run_experiment.py --cfg=experiments/swiss_roll/config/exact_cme_process.yaml --o=sandbox/sandbox/exact_wo_noise --plot
# python experiments/swiss_roll/run_experiment.py --cfg=experiments/swiss_roll/config/exact_cme_process_indiv_noise.yaml --o=sandbox/sandbox/exact_w_noise --plot
# python experiments/swiss_roll/run_experiment.py --cfg=experiments/swiss_roll/config/variational_cme_process.yaml --o=sandbox/sandbox/var_wo_noise --plot --n_epochs=400
# python experiments/swiss_roll/run_experiment.py --cfg=experiments/swiss_roll/config/variational_cme_process_indiv_noise.yaml --o=sandbox/sandbox/var_w_noise --n_epochs=400
# python experiments/swiss_roll/run_experiment.py --cfg=experiments/swiss_roll/config/vbagg.yaml --o=sandbox/sandbox/vbagg --plot
# python experiments/swiss_roll/run_experiment.py --cfg=experiments/swiss_roll/config/gp_regression.yaml --o=sandbox/sandbox/gp_regression --plot
echo "Job Completed"
