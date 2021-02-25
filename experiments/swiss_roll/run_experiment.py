"""
Description : Runs swiss roll experiment

    (1) - Generates bagged swiss roll dataset
    (2) - Fits aggregate model hyperparameters on generated dataset
    (3) - Compute model prediction on indiviuals

Usage: run_experiment.py  --cfg=<path_to_config> --o=<output_dir> [--nbags=<nbags>] [--mean_bag_size=<mean_bag_size>] [--std_bag_size=<std_bag_size>] [--seed=<seed>]

Options:
  --cfg=<path_to_config>             Path to YAML configuration file to use
  --nbags=<nbags>                    Number of bags to generate
  --mean_bag_size=<mean_bag_size>    Mean size of sampled bags
  --std_bag_size=<std_bag_size>      Size standard deviation of sampled bags
  --o=<output_dir>                   Output directory
  --seed=<seed>                      Random seed
"""
import os
import yaml
import logging
from docopt import docopt
import matplotlib.pyplot as plt
import generation as gen
import visualization as vis
from models import build_model, train_model, predict


def main(args, cfg):
    # Generate bagged swiss roll dataset
    bags_sizes, individuals, bags_values, aggregate_targets, X_gt, t_gt = make_dataset(cfg['dataset'])
    _ = vis.plot_dataset(individuals, X_gt, t_gt, aggregate_targets, bags_sizes)
    plt.savefig(os.path.join(args['--o'], 'dataset.png'))
    plt.close()
    logging.info("Generated bag swiss roll dataset\n")

    # Create model
    cfg['model'].update(individuals=individuals,
                        bags_values=bags_values,
                        aggregate_targets=aggregate_targets,
                        bags_sizes=bags_sizes)
    model = build_model(cfg['model'])
    logging.info(f"Initialized model \n{model}\n")

    # Fit hyperparameters
    logging.info("Fitting model hyperparameters")
    cfg['training'].update(model=model,
                           individuals=individuals,
                           bags_values=bags_values,
                           aggregate_targets=aggregate_targets,
                           bags_sizes=bags_sizes)
    train_model(cfg['training'])

    # Compute individuals predictive posterior and plot prediction
    predict_kwargs = {'name': cfg['model']['name'],
                      'model': model,
                      'individuals': X_gt}
    individuals_posterior = predict(predict_kwargs)
    _ = vis.plot_grountruth_prediction(individuals_posterior, X_gt, t_gt)
    plt.savefig(os.path.join(args['--o'], 'prediction.png'))
    plt.close()


def make_dataset(cfg):
    """Generates bagged swiss-roll dataset

    Args:
        cfg (dict): input arguments

    Returns:
        type: list[int], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor

    """
    # Set aside random seed arg
    seed = int(args['--seed']) if args['--seed'] else None

    # Sample bags sizes
    bags_sizes = gen.sample_bags_sizes(mean_bag_size=cfg['mean_bag_size'],
                                       std_bag_size=cfg['std_bag_size'],
                                       n_bags=cfg['nbags'],
                                       seed=cfg['seed'])
    n_samples = sum(bags_sizes)

    # Generate groundtruth and uniform swiss rolls
    X_gt, t_gt = gen.make_swiss_roll(n_samples=cfg['n_samples_groundtruth'],
                                     groundtruth=True,
                                     standardize=True,
                                     seed=seed)
    individuals, _ = gen.make_swiss_roll(n_samples=n_samples,
                                         groundtruth=False,
                                         standardize=True,
                                         seed=seed)

    # Aggregate individuals into bags
    bags_values, bags_heights = gen.aggregate_bags(X=individuals, bags_sizes=bags_sizes)

    # Compute bags aggregate target based on groundtruth
    aggregate_targets = gen.aggregate_targets(X=X_gt, t=t_gt, bags_heights=bags_heights)

    return bags_sizes, individuals, bags_values, aggregate_targets, X_gt, t_gt


def update_cfg(cfg, args):
    """Updates loaded configuration file with specified command line arguments

    Args:
        cfg (dict): loaded configuration file
        args (dict): script execution arguments

    Returns:
        type: dict

    """
    if args['--nbags']:
        cfg['dataset']['nbags'] = int(args['--nbags'])
    if args['--mean_bag_size']:
        cfg['dataset']['mean_bag_size'] = int(args['--mean_bag_size'])
    if args['--std_bag_size']:
        cfg['dataset']['std_bag_size'] = int(args['--std_bag_size'])
    if args['--seed']:
        cfg['dataset']['seed'] = int(args['--seed'])
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

    # Run session
    main(args, cfg)
