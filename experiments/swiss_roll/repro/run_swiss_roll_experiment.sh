# Define configuration files path variables
EXACT_CME_CFG=experiments/swiss_roll/config/exact_cme_process.yaml
VARIATIONAL_CME_CFG=experiments/swiss_roll/config/variational_cme_process.yaml
VBAGG_CFG=experiments/swiss_roll/config/vbagg.yaml
LINEAR_INTERPOLATION_CFG=experiments/swiss_roll/config/linear_interpolation.yaml

# Define output directories path variables
EXACT_CME_OUTDIR=experiments/swiss_roll/data/experiment_outputs/seeds/exact_cme_process
VARIATIONAL_CME_OUTDIR=experiments/swiss_roll/data/experiment_outputs/seeds/variational_cme_process
VBAGG_OUTDIR=experiments/swiss_roll/data/experiment_outputs/seeds/vbagg
LINEAR_INTERPOLATION_OUTDIR=experiments/swiss_roll/data/experiment_outputs/seeds/linear_interpolation

# Run experiments for multiple seeds
for SEED in 3 5 7 13 15 19 23 31 37 41 42 59 61 67 71 73 79 83 97 101 ;
do
  DIRNAME=seed_$SEED
  python experiments/swiss_roll/run_experiment.py --seed=$SEED --cfg=$EXACT_CME_CFG --o=$EXACT_CME_OUTDIR/$DIRNAME
  python experiments/swiss_roll/run_experiment.py --seed=$SEED --cfg=$VARIATIONAL_CME_CFG --o=$VARIATIONAL_CME_OUTDIR/$DIRNAME
  python experiments/swiss_roll/run_experiment.py --seed=$SEED --cfg=$VBAGG_CFG --o=$VBAGG_OUTDIR/$DIRNAME
  python experiments/swiss_roll/run_experiment.py --seed=$SEED --cfg=$LINEAR_INTERPOLATION_CFG --o=$LINEAR_INTERPOLATION_OUTDIR/$DIRNAME
done
