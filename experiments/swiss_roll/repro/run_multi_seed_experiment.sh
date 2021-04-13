# Define configuration files path variables
EXACT_CME_CFG=experiments/swiss_roll/config/exact_cme_process.yaml
EXACT_CME_INDIV_NOISE_CFG=experiments/swiss_roll/config/exact_cme_process_indiv_noise.yaml
VARIATIONAL_CME_CFG=experiments/swiss_roll/config/variational_cme_process.yaml
VARIATIONAL_CME_INDIV_NOISE_CFG=experiments/swiss_roll/config/variational_cme_process_indiv_noise.yaml
VBAGG_CFG=experiments/swiss_roll/config/vbagg.yaml
GP_REGRESSION_CFG=experiments/swiss_roll/config/gp_regression.yaml

# Define output directories path variables
EXACT_CME_OUTDIR=experiments/swiss_roll/data/experiment_outputs/seeds/exact_cme_process
EXACT_CME_INDIV_NOISE_OUTDIR=experiments/swiss_roll/data/experiment_outputs/seeds/exact_cme_process_indiv_noise
VARIATIONAL_CME_OUTDIR=experiments/swiss_roll/data/experiment_outputs/seeds/variational_cme_process
VARIATIONAL_CME_INDIV_NOISE_OUTDIR=experiments/swiss_roll/data/experiment_outputs/seeds/variational_cme_process_indiv_noise
RAW_VBAGG_OUTDIR=experiments/swiss_roll/data/experiment_outputs/seeds/vbagg_beta_1
VBAGG_OUTDIR=experiments/swiss_roll/data/experiment_outputs/seeds/vbagg
GP_REGRESSION_OUTDIR=experiments/swiss_roll/data/experiment_outputs/seeds/gp_regression

# Run experiments for multiple seeds
for SEED in 3 5 7 13 15 19 23 31 37 41 42 59 61 67 71 73 79 83 97 101 ;
do
  DIRNAME=seed_$SEED
  python experiments/swiss_roll/run_experiment.py --seed=$SEED --cfg=$EXACT_CME_CFG --o=$EXACT_CME_OUTDIR/$DIRNAME
  python experiments/swiss_roll/run_experiment.py --seed=$SEED --cfg=$EXACT_CME_INDIV_NOISE_CFG --o=$EXACT_CME_INDIV_NOISE_OUTDIR/$DIRNAME
  python experiments/swiss_roll/run_experiment.py --seed=$SEED --cfg=$VARIATIONAL_CME_CFG --o=$VARIATIONAL_CME_OUTDIR/$DIRNAME
  python experiments/swiss_roll/run_experiment.py --seed=$SEED --cfg=$VARIATIONAL_CME_INDIV_NOISE_CFG --o=$VARIATIONAL_CME_INDIV_NOISE_OUTDIR/$DIRNAME
  python experiments/swiss_roll/run_experiment.py --seed=$SEED --cfg=$VBAGG_CFG --o=$RAW_VBAGG_OUTDIR/$DIRNAME --beta=1
  python experiments/swiss_roll/run_experiment.py --seed=$SEED --cfg=$VBAGG_CFG --o=$VBAGG_OUTDIR/$DIRNAME
  python experiments/swiss_roll/run_experiment.py --seed=$SEED --cfg=$GP_REGRESSION_CFG --o=$GP_REGRESSION_OUTDIR/$DIRNAME
done
