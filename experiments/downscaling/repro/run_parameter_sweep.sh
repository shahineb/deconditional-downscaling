# Define configuration files path variables
VARIATIONAL_CME_CFG=experiments/downscaling/config/variational_cme_process.yaml
VARIATIONAL_CME_INDIV_NOISE_CFG=experiments/downscaling/config/variational_cme_process_indiv_noise.yaml
VBAGG_CFG=experiments/downscaling/config/vbagg.yaml
KRIGGING_CFG=experiments/downscaling/config/krigging.yaml

# Define output directories path variables
VARIATIONAL_CME_OUTDIR=experiments/downscaling/data/experiment_outputs/parameter_sweep/variational_cme_process
VARIATIONAL_CME_INDIV_NOISE_OUTDIR=experiments/downscaling/data/experiment_outputs/parameter_sweep/variational_cme_process_indiv_noise
VBAGG_OUTDIR=experiments/downscaling/data/experiment_outputs/parameter_sweep/vbagg
KRIGGING_OUTDIR=experiments/downscaling/data/experiment_outputs/parameter_sweep/krigging

# Define parameter grid to parse onto
VALUES_LR=(3e-2 3e-3 3e-4)
VALUES_BETA=(1e-1 1e-2 1e-3)
VALUES_LBDA=(1e-1 1e-2 1e-3 1e-4 1e-5)



# # Variational CMP w/o individuals noise
# for lbda in ${VALUES_LBDA[@]}
# do
#   output_dir="lbda_"$lbda
#   python experiments/downscaling/run_experiment.py --cfg=$VARIATIONAL_CME_CFG --o=$VARIATIONAL_CME_OUTDIR/$output_dir \
#   --lbda=$lbda
# done
#
# for beta in ${VALUES_BETA[@]}
# do
#   output_dir="beta_"$beta
#   python experiments/downscaling/run_experiment.py --cfg=$VARIATIONAL_CME_CFG --o=$VARIATIONAL_CME_OUTDIR/$output_dir \
#   --beta=$beta
# done



# Variational CMP w/ individuals noise
for lbda in ${VALUES_LBDA[@]}
do
  output_dir="lbda_"$lbda
  python experiments/downscaling/run_experiment.py --cfg=$VARIATIONAL_CME_INDIV_NOISE_CFG --o=$VARIATIONAL_CME_INDIV_NOISE_OUTDIR/$output_dir \
  --lbda=$lbda
done

for beta in ${VALUES_BETA[@]}
do
  output_dir="beta_"$beta
  python experiments/downscaling/run_experiment.py --cfg=$VARIATIONAL_CME_INDIV_NOISE_CFG --o=$VARIATIONAL_CME_INDIV_NOISE_OUTDIR/$output_dir \
  --beta=$beta
done

for lr in ${VALUES_LR[@]}
do
  output_dir="lr_"$lr
  python experiments/downscaling/run_experiment.py --cfg=$VARIATIONAL_CME_INDIV_NOISE_CFG --o=$VARIATIONAL_CME_INDIV_NOISE_OUTDIR/$output_dir \
  --lr=$lr
done


# VbAgg
for beta in ${VALUES_BETA[@]}
do
  output_dir="beta_"$beta
  python experiments/downscaling/run_experiment.py --cfg=$VBAGG_CFG --o=$VBAGG_OUTDIR/$output_dir \
  --beta=$beta
done


# Krigging
for beta in ${VALUES_BETA[@]}
do
  output_dir="beta_"$beta
  python experiments/downscaling/run_experiment.py --cfg=$KRIGGING_CFG --o=$KRIGGING_OUTDIR/$output_dir \
  --beta=$beta
done
