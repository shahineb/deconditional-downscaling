# Define configuration files path variables
VARIATIONAL_CME_CFG=experiments/downscaling/config/variational_cme_process.yaml

# Define output directories path variables
VARIATIONAL_CME_OUTDIR=experiments/downscaling/data/experiment_outputs/coarseness/variational_cme_process

# Run experiments for block size multiple seeds
for SEED in 3 5 7 ;
do
  DIRNAME=block_size_3/seed_$SEED
  python experiments/downscaling/run_experiment.py --plot --seed=$SEED --block_size=3 --batch_size=256 --cfg=$VARIATIONAL_CME_CFG --o=$VARIATIONAL_CME_OUTDIR/$DIRNAME

  DIRNAME=block_size_5/seed_$SEED
  python experiments/downscaling/run_experiment.py --plot --seed=$SEED --block_size=5 --batch_size=92 --cfg=$VARIATIONAL_CME_CFG --o=$VARIATIONAL_CME_OUTDIR/$DIRNAME

  DIRNAME=block_size_7/seed_$SEED
  python experiments/downscaling/run_experiment.py --plot --seed=$SEED --block_size=7 --batch_size=47 --cfg=$VARIATIONAL_CME_CFG --o=$VARIATIONAL_CME_OUTDIR/$DIRNAME

  DIRNAME=block_size_9/seed_$SEED
  python experiments/downscaling/run_experiment.py --plot --seed=$SEED --block_size=9 --batch_size=29 --cfg=$VARIATIONAL_CME_CFG --o=$VARIATIONAL_CME_OUTDIR/$DIRNAME

  DIRNAME=block_size_11/seed_$SEED
  python experiments/downscaling/run_experiment.py --plot --seed=$SEED --block_size=11 --batch_size=19 --cfg=$VARIATIONAL_CME_CFG --o=$VARIATIONAL_CME_OUTDIR/$DIRNAME

  DIRNAME=block_size_13/seed_$SEED
  python experiments/downscaling/run_experiment.py --plot --seed=$SEED --block_size=13 --batch_size=14 --cfg=$VARIATIONAL_CME_CFG --o=$VARIATIONAL_CME_OUTDIR/$DIRNAME

  DIRNAME=block_size_15/seed_$SEED
  python experiments/downscaling/run_experiment.py --plot --seed=$SEED --block_size=15 --batch_size=10 --cfg=$VARIATIONAL_CME_CFG --o=$VARIATIONAL_CME_OUTDIR/$DIRNAME
done
