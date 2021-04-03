# Define configuration files path variables
VARIATIONAL_CME_CFG=experiments/downscaling/config/variational_cme_process.yaml

# Define output directories path variables
VARIATIONAL_CME_OUTDIR=sandbox/dummy_downscaling

# Run experiments for block size multiple seeds
for SEED in 3 ;
do
  DIRNAME=block_size_3/seed_$SEED
  python experiments/downscaling/run_experiment.py --plot --seed=$SEED --block_size=3 --batch_size=256 --cfg=$VARIATIONAL_CME_CFG --o=$VARIATIONAL_CME_OUTDIR/$DIRNAME
done
