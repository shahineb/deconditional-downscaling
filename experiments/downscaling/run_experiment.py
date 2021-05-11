"""
Description : Runs statistical downscaling experiment

    (1) - Loads in and preprocesses cloud fields
    (2) - Fits aggregate model hyperparameters on dataset
    (3) - Compute model prediction on indiviuals

Usage: run_experiment.py  [options] --cfg=<path_to_config> --o=<output_dir>

Options:
  --cfg=<path_to_config>           Path to YAML configuration file to use.
  --o=<output_dir>                 Output directory.
  --device=<device_index>          Index of GPU to use [default: 0].
  --block_size=<block_size>        Dimension of block to use for groundtruth downsampling.
  --batch_size=<batch_size>        Batch size used for stochastic optimization scheme
  --lr=<lr>                        Learning rate
  --beta=<beta>                    Weight of KL term in ELBO
  --lbda=<lbda>                    CME inverse regularization term
  --n_epochs=<n_epochs>            Number of training epochs
  --plot                           Outputs scatter plots.
  --seed=<seed>                    Random seed.
"""
import os
import yaml
import logging
from docopt import docopt
import xarray as xr
import core.preprocessing as preproc
from models import build_model, train_model


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
    logging.info("Initializing model")
    cfg['model'].update(covariates_grid=covariates_grid)
    model = build_model(cfg['model'])
    logging.info(f"{model}\n")

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
                           device_idx=args['--device'],
                           plot=args['--plot'],
                           dump_dir=args['--o'])
    train_model(cfg['training'])


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
    if args['--n_epochs']:
        cfg['training']['n_epochs'] = int(args['--n_epochs'])
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
