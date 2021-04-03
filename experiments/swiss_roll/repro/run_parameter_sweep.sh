# Define configuration files path variables
EXACT_CME_CFG=experiments/swiss_roll/config/exact_cme_process.yaml
VARIATIONAL_CME_CFG=experiments/swiss_roll/config/variational_cme_process.yaml
VBAGG_CFG=experiments/swiss_roll/config/vbagg.yaml

# Define output directories path variables
EXACT_CME_OUTDIR=experiments/swiss_roll/data/experiment_outputs/parameter_sweep/exact_cme_process
VARIATIONAL_CME_OUTDIR=experiments/swiss_roll/data/experiment_outputs/parameter_sweep/variational_cme_process
VBAGG_OUTDIR=experiments/swiss_roll/data/experiment_outputs/parameter_sweep/vbagg


# Exact CME
for lbda in 0.001 0.002 0.003 0.004 0.005 0.006 0.007 0.008 0.009 0.01 0.02 0.03 0.04 0.05 0.06 0.08 0.1;
do
  output_dir="lbda_"$lbda
  python experiments/swiss_roll/run_experiment.py --cfg=$EXACT_CME_CFG --o=$EXACT_CME_OUTDIR/$output_dir \
  --lbda=$lbda
done

# Variational CME
for lbda in 0.00001 0.00003 0.00002 0.00004 0.00005 0.00006 0.00007 0.00008 0.00009 0.0001 0.0002 0.0003 0.0004 0.0005 0.0006 0.0007 0.0008 0.0009 0.001;
do
  output_dir="lbda_"$lbda
  python experiments/swiss_roll/run_experiment.py --cfg=$VARIATIONAL_CME_CFG --o=$VARIATIONAL_CME_OUTDIR/$output_dir \
  --lbda=$lbda
done

for beta in 0.0001 0.0002 0.0003 0.0004 0.0005 0.0006 0.0007 0.0008 0.0009 0.001 0.002 0.003 0.004 0.005 0.006 0.007 0.008 0.009 0.01 1;
do
  output_dir="beta_"$beta"_lbda_"$lbda
  python experiments/swiss_roll/run_experiment.py --cfg=$VARIATIONAL_CME_CFG --o=$VARIATIONAL_CME_OUTDIR/$output_dir \
  --beta=$beta
done

# VBAgg
for beta in 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1;
do
  output_dir="beta_"$beta"_lbda_"$lbda
  python experiments/swiss_roll/run_experiment.py --cfg=$VBAGG_CFG --o=$VBAGG_OUTDIR/$output_dir \
  --beta=$beta
done
