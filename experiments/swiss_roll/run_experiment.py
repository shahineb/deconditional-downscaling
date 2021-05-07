"""
Description : Runs swiss roll experiment

    (1) - Generates bagged swiss roll dataset
    (2) - Fits aggregate model hyperparameters on generated dataset
    (3) - Compute model prediction on indiviuals

Usage: run_experiment.py  [options] --cfg=<path_to_config> --o=<output_dir>

Options:
  --cfg=<path_to_config>           Path to YAML configuration file to use.
  --n_bags=<n_bags>                Number of bags to generate.
  --o=<output_dir>                 Output directory.
  --lr=<lr>                        Learning rate.
  --beta=<beta>                    Weight of KL term in ELBO for variational formulation.
  --lbda=<lbda>                    CME inverse regularization term.
  --n_epochs=<n_epochs>            Number of training epochs
  --plot                           Outputs scatter plots.
  --seed=<seed>                    Random seed.
"""
import os
import yaml
import logging
from docopt import docopt
import matplotlib.pyplot as plt
import core.generation as gen
import core.visualization as vis
from models import build_model, train_model, predict


def main(args, cfg):
    # Generate bagged swiss roll dataset
    bags_sizes, individuals, bags_values, aggregate_targets, X, t, X_gt, t_gt = make_dataset(cfg['dataset'])
    logging.info("Generated bag swiss roll dataset\n")

    # Save dataset scatter plot
    if args['--plot']:
        dump_plot(plotting_function=vis.plot_dataset,
                  filename='dataset.png',
                  output_dir=args['--o'],
                  individuals=individuals,
                  groundtruth_individuals=X_gt,
                  targets=t_gt,
                  aggregate_targets=aggregate_targets,
                  bags_sizes=bags_sizes)

    # Create model
    cfg['model'].update(individuals=individuals,
                        bags_values=bags_values,
                        aggregate_targets=aggregate_targets,
                        bags_sizes=bags_sizes)
    model = build_model(cfg['model'])
    logging.info(f"Initialized model \n{model}\n")

    # Fit hyperparameters
    logging.info("Fitting model hyperparameters\n")
    cfg['training'].update(model=model,
                           individuals=individuals,
                           bags_values=bags_values,
                           aggregate_targets=aggregate_targets,
                           bags_sizes=bags_sizes,
                           groundtruth_individuals=X,
                           groundtruth_targets=t,
                           chunk_size=cfg['evaluation']['chunk_size_nll'],
                           dump_dir=args['--o'])
    train_model(cfg['training'])

    # Save prediction scatter plot
    if args['--plot']:
        # Compute individuals poserior on groundtruth distorted swiss roll
        logging.info("Plotting individuals posterior\n")
        predict_kwargs = {'name': cfg['model']['name'],
                          'model': model.eval().cpu(),
                          'individuals': X}
        individuals_posterior = predict(predict_kwargs)

        # Dump scatter plot
        dump_plot(plotting_function=vis.plot_grountruth_prediction,
                  filename='prediction.png',
                  output_dir=args['--o'],
                  individuals_posterior=individuals_posterior,
                  groundtruth_individuals=X,
                  targets=t)


def make_dataset(cfg):
    """Generates bagged swiss-roll dataset

    Args:
        cfg (dict): input arguments

    Returns:
        type: list[int], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor

    """
    # Generate groundtruth and uniform swiss rolls
    X_gt, t_gt = gen.make_swiss_roll(n_samples=cfg['n_samples'],
                                     groundtruth=True,
                                     standardize=True,
                                     seed=cfg['seed'])
    X, t = gen.make_swiss_roll(n_samples=cfg['n_samples'],
                               groundtruth=False,
                               standardize=True,
                               seed=cfg['seed'])

    # Compose into bagged dataset
    individuals, bags_sizes, bags_values, aggregate_targets = gen.compose_bags_dataset(X=X,
                                                                                       X_gt=X_gt,
                                                                                       t_gt=t_gt,
                                                                                       n_bags=cfg['n_bags'],
                                                                                       noise=cfg['noise'])

    return bags_sizes, individuals, bags_values, aggregate_targets, X, t, X_gt, t_gt


def dump_plot(plotting_function, filename, output_dir, *plot_args, **plot_kwargs):
    """Plot dumping utility

    Args:
        plotting_function (callable): plotting utility
        filename (str): name of saved png file
        output_dir (str): directory where file is dumped

    """
    _ = plotting_function(*plot_args, **plot_kwargs)
    dump_path = os.path.join(output_dir, filename)
    plt.savefig(dump_path)
    plt.close()
    logging.info(f"Plot saved at {dump_path}\n")


def update_cfg(cfg, args):
    """Updates loaded configuration file with specified command line arguments

    Args:
        cfg (dict): loaded configuration file
        args (dict): script execution arguments

    Returns:
        type: dict

    """
    if args['--n_bags']:
        cfg['dataset']['n_bags'] = int(args['--n_bags'])
    if args['--seed']:
        cfg['dataset']['seed'] = int(args['--seed'])
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
    with open(os.path.join(args['--o'], 'cfg.yaml'), 'w') as f:
        yaml.dump(cfg, f)

    # Run session
    main(args, cfg)
