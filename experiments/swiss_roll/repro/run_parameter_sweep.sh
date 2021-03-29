# Define configuration files path variables
EXACT_CME_CFG=experiments/swiss_roll/config/exact_cme_process.yaml
VARIATIONAL_CME_CFG=experiments/swiss_roll/config/variational_cme_process.yaml
VBAGG_CFG=experiments/swiss_roll/config/vbagg.yaml

# Define output directories path variables
EXACT_CME_OUTDIR=experiments/swiss_roll/data/experiment_outputs/parameter_sweep/exact_cme_process
VARIATIONAL_CME_OUTDIR=experiments/swiss_roll/data/experiment_outputs/parameter_sweep/variational_cme_process
VBAGG_OUTDIR=experiments/swiss_roll/data/experiment_outputs/parameter_sweep/vbagg

# Define parameter grid to parse onto
VALUES_BETA=(1 1e-3 1e-6)
VALUES_LBDA=(1e-1 1e-3 1e-5)

# Exact CME
for lbda in ${VALUES_LBDA[@]}
do
  output_dir="beta_"$beta"_lbda_"$lbda
  python experiments/swiss_roll/run_experiment.py --cfg=$EXACT_CME_CFG --o=$EXACT_CME_OUTDIR/$output_dir \
  --lbda=$lbda
done

# Variational CME
for beta in ${VALUES_BETA[@]}
do
  for lbda in ${VALUES_LBDA[@]}
  do
    output_dir="beta_"$beta"_lbda_"$lbda
    python experiments/swiss_roll/run_experiment.py --cfg=$VARIATIONAL_CME_CFG --o=$VARIATIONAL_CME_OUTDIR/$output_dir \
    --beta=$beta --lbda=$lbda
  done
done

# VBAgg
for beta in ${VALUES_BETA[@]}
do
  for lbda in ${VALUES_LBDA[@]}
  do
    output_dir="beta_"$beta"_lbda_"$lbda
    python experiments/swiss_roll/run_experiment.py --cfg=$VBAGG_CFG --o=$VBAGG_OUTDIR/$output_dir \
    --beta=$beta --lbda=$lbda
  done
done
