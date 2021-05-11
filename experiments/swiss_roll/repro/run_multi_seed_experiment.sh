# Define configuration files path variables
EXACT_CME_CFG=experiments/swiss_roll/config/exact_cme_process.yaml
EXACT_CME_INDIV_NOISE_CFG=experiments/swiss_roll/config/exact_cme_process_indiv_noise.yaml
BAGGED_GP_CFG=experiments/swiss_roll/config/bagged_gp.yaml
INDEPENDENT_BAGGED_GP_CFG=experiments/swiss_roll/config/bagged_gp_independent_bags.yaml
VARIATIONAL_CME_CFG=experiments/swiss_roll/config/variational_cme_process.yaml
VARIATIONAL_CME_INDIV_NOISE_CFG=experiments/swiss_roll/config/variational_cme_process_indiv_noise.yaml
VBAGG_CFG=experiments/swiss_roll/config/vbagg.yaml
GP_REGRESSION_CFG=experiments/swiss_roll/config/gp_regression.yaml

# Define output directories path variables
EXACT_CME_OUTDIR=experiments/swiss_roll/data/experiment_outputs/seeds/exact_cme_process
EXACT_CME_INDIV_NOISE_OUTDIR=experiments/swiss_roll/data/experiment_outputs/seeds/exact_cme_process_indiv_noise
BAGGED_GP_OUTDIR=experiments/swiss_roll/data/experiment_outputs/seeds/bagged_gp
INDEPENDENT_BAGGED_GP_OUTDIR=experiments/swiss_roll/data/experiment_outputs/seeds/bagged_gp_independent_bags
VARIATIONAL_CME_OUTDIR=experiments/swiss_roll/data/experiment_outputs/seeds/variational_cme_process
VARIATIONAL_CME_INDIV_NOISE_OUTDIR=experiments/swiss_roll/data/experiment_outputs/seeds/variational_cme_process_indiv_noise
VBAGG_OUTDIR=experiments/swiss_roll/data/experiment_outputs/seeds/vbagg
GP_REGRESSION_OUTDIR=experiments/swiss_roll/data/experiment_outputs/seeds/gp_regression

# Run experiments for multiple seeds
for SEED in 2 3 5 7 9 11 17 19 23 29 31 37 41 42;
do
  DIRNAME=seed_$SEED
  # python experiments/swiss_roll/run_experiment.py --seed=$SEED --cfg=$EXACT_CME_CFG --o=$EXACT_CME_OUTDIR/$DIRNAME
  # python experiments/swiss_roll/run_experiment.py --seed=$SEED --cfg=$EXACT_CME_INDIV_NOISE_CFG --o=$EXACT_CME_INDIV_NOISE_OUTDIR/$DIRNAME
  # python experiments/swiss_roll/run_experiment.py --seed=$SEED --cfg=$BAGGED_GP_CFG --o=$BAGGED_GP_OUTDIR/$DIRNAME
  python experiments/swiss_roll/run_experiment.py --seed=$SEED --cfg=$INDEPENDENT_BAGGED_GP_CFG --o=$INDEPENDENT_BAGGED_GP_OUTDIR/$DIRNAME --device=3
  # python experiments/swiss_roll/run_experiment.py --seed=$SEED --cfg=$VARIATIONAL_CME_CFG --o=$VARIATIONAL_CME_OUTDIR/$DIRNAME
  # python experiments/swiss_roll/run_experiment.py --seed=$SEED --cfg=$VARIATIONAL_CME_INDIV_NOISE_CFG --o=$VARIATIONAL_CME_INDIV_NOISE_OUTDIR/$DIRNAME
  # python experiments/swiss_roll/run_experiment.py --seed=$SEED --cfg=$VBAGG_CFG --o=$VBAGG_OUTDIR/$DIRNAME
  # python experiments/swiss_roll/run_experiment.py --seed=$SEED --cfg=$GP_REGRESSION_CFG --o=$GP_REGRESSION_OUTDIR/$DIRNAME
done
