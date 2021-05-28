# Define configuration files path variables
VARIATIONAL_CME_INDIV_NOISE_CFG=experiments/downscaling/config/variational_cme_process_indiv_noise.yaml
VBAGG_CFG=experiments/downscaling/config/vbagg.yaml
KRIGGING_CFG=experiments/downscaling/config/krigging.yaml


# Define output directories path variables
VARIATIONAL_CME_INDIV_NOISE_OUTDIR=experiments/downscaling/data/experiment_outputs/seeds/variational_cme_process_indiv_noise
VBAGG_OUTDIR=experiments/downscaling/data/experiment_outputs/seeds/vbagg
KRIGGING_OUTDIR=experiments/downscaling/data/experiment_outputs/seeds/krigging


# Run experiments for multiple seeds
for SEED in 2 3 5 7 11 13 17 19 23 42 ;
do
  DIRNAME=seed_$SEED
  python experiments/downscaling/run_experiment.py --seed=$SEED --cfg=$VARIATIONAL_CME_INDIV_NOISE_CFG --o=$VARIATIONAL_CME_INDIV_NOISE_OUTDIR/$DIRNAME --plot
  python experiments/downscaling/run_experiment.py --seed=$SEED --cfg=$VBAGG_CFG --o=$VBAGG_OUTDIR/$DIRNAME
  python experiments/downscaling/run_experiment.py --seed=$SEED --cfg=$KRIGGING_CFG --o=$KRIGGING_OUTDIR/$DIRNAME
done
