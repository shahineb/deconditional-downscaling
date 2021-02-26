# Define configuration files path variables
EXACT_CME_CFG=experiments/swiss_roll/config/exact_cme_process.yaml
VARIATIONAL_CME_CFG=experiments/swiss_roll/config/variational_cme_process.yaml
VBAGG_CFG=experiments/swiss_roll/config/vbagg.yaml
LINEAR_INTERPOLATION_CFG=experiments/swiss_roll/config/linear_interpolation.yaml

# Define output directories path variables
EXACT_CME_OUTDIR=experiments/swiss_roll/data/experiment_outputs/exact_cme_process
VARIATIONAL_CME_OUTDIR=experiments/swiss_roll/data/experiment_outputs/variational_cme_process
VBAGG_OUTDIR=experiments/swiss_roll/data/experiment_outputs/vbagg
LINEAR_INTERPOLATION_OUTDIR=experiments/swiss_roll/data/experiment_outputs/linear_interpolation

# Run experiments for multiple seeds
for SEED in 5 13 42 73 101 ;
do
  DIRNAME=seed_$SEED
  python experiments/swiss_roll/run_experiment.py --plot --seed=$SEED --cfg=$EXACT_CME_CFG --o=$EXACT_CME_OUTDIR/$DIRNAME
  python experiments/swiss_roll/run_experiment.py --plot --seed=$SEED --cfg=$VARIATIONAL_CME_CFG --o=$VARIATIONAL_CME_OUTDIR/$DIRNAME
  python experiments/swiss_roll/run_experiment.py --plot --seed=$SEED --cfg=$VBAGG_CFG --o=$VBAGG_OUTDIR/$DIRNAME
  python experiments/swiss_roll/run_experiment.py --plot --seed=$SEED --cfg=$LINEAR_INTERPOLATION_CFG --o=$LINEAR_INTERPOLATION_OUTDIR/$DIRNAME
done
