# Define configuration files path variables
VARIATIONAL_CME_CFG=experiments/downscaling/config/variational_cme_process.yaml

# Define output directories path variables
VARIATIONAL_CME_OUTDIR=experiments/downscaling/data/experiment_outputs/coarseness/variational_cme_process

# Run experiments for block size multiple seeds
for BLOCK_SIZE in 3 5 7 9 11 13;
do
  for SEED in 3 5 7 ;
  do
    DIRNAME=block_size_$BLOCK_SIZE/seed_$SEED
    python experiments/downscaling/run_experiment.py --plot --seed=$SEED --block_size=$BLOCK_SIZE --cfg=$VARIATIONAL_CME_CFG --o=$VARIATIONAL_CME_OUTDIR/$DIRNAME
  done
done
