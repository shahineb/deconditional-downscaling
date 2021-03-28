# Define configuration files path variables
VARIATIONAL_CME_CFG=experiments/downscaling/config/variational_cme_process.yaml

# Define output directories path variables
VARIATIONAL_CME_OUTDIR=experiments/downscaling/data/experiment_outputs/parameter_sweep/variational_cme_process

# Define parameter grid to parse onto
VALUES_BATCHSIZE=(8 32 128)
VALUES_BETA=(1 1e-3 1e-6)
VALUES_LBDA=(1e-1 1e-3 1e-5)
VALUES_LR=(1e-2 1e-3 1e-4)

# Iterate over all possible values
for batch_size in ${VALUES_BATCHSIZE[@]}
do
  for beta in ${VALUES_BETA[@]}
  do
    for lbda in ${VALUES_LBDA[@]}
    do
      for lr in ${VALUES_LR[@]}
      do
        output_dir="batch_size_"$batch_size"_beta_"$beta"_lbda_"$lbda"_lr_"$lr
        python experiments/downscaling/run_experiment.py --cfg=$VARIATIONAL_CME_CFG --o=$VARIATIONAL_CME_OUTDIR/$output_dir \
        --batch_size=$batch_size --beta=$beta --lbda=$lbda --lr=$lr
      done
    done
  done
done
