# Define configuration files path variables
EXACT_CME_CFG=experiments/swiss_roll/config/exact_cme_process.yaml
BAGG_GP_CFG=experiments/swiss_roll/config/bagged_gp.yaml
VARIATIONAL_CME_CFG=experiments/swiss_roll/config/variational_cme_process.yaml
VBAGG_CFG=experiments/swiss_roll/config/vbagg.yaml
GP_REGRESSION_CFG=experiments/swiss_roll/config/gp_regression.yaml

# Define output directories path variables
EXACT_CME_OUTDIR=experiments/swiss_roll/data/experiment_outputs/seeds/unmatched/exact_cme_process
BAGG_GP_OUTDIR=experiments/swiss_roll/data/experiment_outputs/seeds/unmatched/bagg_gp
VARIATIONAL_CME_OUTDIR=experiments/swiss_roll/data/experiment_outputs/seeds/unmatched/variational_cme_process
VBAGG_OUTDIR=experiments/swiss_roll/data/experiment_outputs/seeds/unmatched/vbagg
GP_REGRESSION_OUTDIR=experiments/swiss_roll/data/experiment_outputs/seeds/unmatched/gp_regression

# Run experiments for multiple seeds
for SEED in 2 3 5 7 9 11 17 19 29 31 41 42 43 47 59 73 79 89 97 101;
do
  DIRNAME=seed_$SEED
  # python experiments/swiss_roll/run_experiment.py --seed=$SEED --cfg=$EXACT_CME_CFG --o=$EXACT_CME_OUTDIR/$DIRNAME --device=4 --unmatched
  python experiments/swiss_roll/run_experiment.py --seed=$SEED --cfg=$BAGG_GP_CFG --o=$BAGG_GP_OUTDIR/$DIRNAME --unmatched --device=2
  # python experiments/swiss_roll/run_experiment.py --seed=$SEED --cfg=$VARIATIONAL_CME_CFG --o=$VARIATIONAL_CME_OUTDIR/$DIRNAME --device=4 --unmatched --lr=0.01 --n_epochs=400
  # python experiments/swiss_roll/run_experiment.py --seed=$SEED --cfg=$VBAGG_CFG --o=$VBAGG_OUTDIR/$DIRNAME --device=4 --unmatched --lr=0.01 --n_epochs=400
  # python experiments/swiss_roll/run_experiment.py --seed=$SEED --cfg=$GP_REGRESSION_CFG --o=$GP_REGRESSION_OUTDIR/$DIRNAME --device=4 --unmatched --lr=0.01 --n_epochs=400
done
