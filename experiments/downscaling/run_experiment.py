"""
Description : Runs statistical downscaling experiment

    (1) - Loads in and preprocesses cloud fields
    (2) - Fits aggregate model hyperparameters on dataset
    (3) - Compute model prediction on indiviuals

Usage: run_experiment.py  [options] --cfg=<path_to_config> --o=<output_dir>

Options:
  --cfg=<path_to_config>           Path to YAML configuration file to use.
  --o=<output_dir>                 Output directory.
  --block_size=<block_size>        Dimension of block to use for groundtruth downsampling.
  --batch_size=<batch_size>        Batch size used for stochastic optimization scheme
  --lr=<lr>                        Learning rate
  --beta=<beta>                    Weight of KL term in ELBO
  --lbda=<lbda>                    CME inverse regularization term
  --plot                           Outputs scatter plots.
  --seed=<seed>                    Random seed.
"""
import os
import yaml
import logging
from docopt import docopt
import xarray as xr
import core.preprocessing as preproc
from core.metrics import compute_metrics
from models import build_model, train_model, predict


def main(args, cfg):
    # Load in CMIP cloud fields
    fields = xr.load_dataset(cfg['dataset']['path'])
    groundtruth_field = preproc.trim(field=fields[cfg['dataset']['target_field_name']], block_size=cfg['dataset']['block_size'])

    # Preprocess into covariates and target fields
    covariates_fields, aggregate_target_field, raw_aggregate_target_field = preproc.preprocess_fields(fields=fields,
                                                                                                      covariate_fields_names=cfg['dataset']['covariate_fields_names'],
                                                                                                      target_field_name=cfg['dataset']['target_field_name'],
                                                                                                      block_size=cfg['dataset']['block_size'])

    # Convert into pytorch friendly dataset
    covariates_grid, covariates_blocks, bags_blocks, extended_bags, targets_blocks = preproc.make_tensor_dataset(covariates_fields=covariates_fields,
                                                                                                                 aggregate_target_field=aggregate_target_field,
                                                                                                                 block_size=cfg['dataset']['block_size'])
    logging.info("Loaded and formatted cloud fields downscaling dataset\n")

    # Create model
    cfg['model'].update(covariates_blocks=covariates_blocks)
    model = build_model(cfg['model'])
    logging.info(f"Initialized model \n{model}\n")

    # Fit hyperparameters
    logging.info("Fitting model hyperparameters\n")
    cfg['training'].update(model=model,
                           covariates_blocks=covariates_blocks,
                           bags_blocks=bags_blocks,
                           extended_bags=extended_bags,
                           targets_blocks=targets_blocks,
                           covariates_grid=covariates_grid,
                           groundtruth_field=groundtruth_field,
                           target_field=raw_aggregate_target_field,
                           step_size=cfg['evaluation']['step_size'],
                           plot=args['--plot'],
                           dump_dir=args['--o'])
    train_model(cfg['training'])

    # Compute individuals predictive posterior and plot prediction
    logging.info("Predicting individuals posterior\n")
    predict_kwargs = {'name': cfg['model']['name'],
                      'model': model,
                      'covariates_grid': covariates_grid,
                      'target_field': raw_aggregate_target_field,
                      'step_size': cfg['evaluation']['step_size']}
    individuals_posterior = predict(predict_kwargs)

    # Evaluate mean metrics
    logging.info("Evaluating model\n")
    evaluate_model(individuals_posterior=individuals_posterior,
                   groundtruth_field=groundtruth_field,
                   output_dir=args['--o'])


def evaluate_model(individuals_posterior, groundtruth_field, output_dir):
    """Computes MSE, MAE, Mean Bias, Spearman Correlation and SSIM between downscaled
    field and groundtruth and dump into YAML metrics file
    """
    individuals_metrics = compute_metrics(individuals_posterior, groundtruth_field)
    dump_path = os.path.join(output_dir, 'metrics.yaml')
    with open(dump_path, 'w') as f:
        yaml.dump(individuals_metrics, f)
    logging.info(f"Metrics : {individuals_metrics}\n")


def update_cfg(cfg, args):
    """Updates loaded configuration file with specified command line arguments

    Args:
        cfg (dict): loaded configuration file
        args (dict): script execution arguments

    Returns:
        type: dict

    """
    if args['--seed']:
        cfg['model']['seed'] = int(args['--seed'])
        cfg['training']['seed'] = int(args['--seed'])
    if args['--block_size']:
        bs = int(args['--block_size'])
        cfg['dataset']['block_size'] = (bs, bs)
    if args['--batch_size']:
        cfg['training']['batch_size'] = int(args['--batch_size'])
    if args['--lbda']:
        cfg['model']['lbda'] = float(args['--lbda'])
    if args['--lr']:
        cfg['training']['lr'] = float(args['--lr'])
    if args['--beta']:
        cfg['training']['beta'] = float(args['--beta'])
    return cfg


if __name__ == "__main__":
    # Read input args
    args = docopt(__doc__)

    # Load config file
    with open(args['--cfg'], "r") as f:
        cfg = yaml.safe_load(f)
    cfg = update_cfg(cfg, args)

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logging.info(f'Arguments: {args}\n')
    logging.info(f'Configuration file: {cfg}\n')

    # Create output directory if doesn't exists
    os.makedirs(args['--o'], exist_ok=True)
    if args['--plot']:
        os.makedirs(os.path.join(args['--o'], 'png'), exist_ok=True)
    with open(os.path.join(args['--o'], 'cfg.yaml'), 'w') as f:
        yaml.dump(cfg, f)

    # Run session
    main(args, cfg)
