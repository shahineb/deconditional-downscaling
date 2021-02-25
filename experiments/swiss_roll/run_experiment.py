"""
Description : Runs swiss roll experiment

    (1) - Generates bagged swiss roll dataset
    (2) - Fits aggregate model hyperparameters on generated dataset
    (3) - Compute model prediction on indiviuals

Usage: run_experiment.py --model=<model_name> --nbags=<nbags> --mean_bag_size=<mean_bag_size> --std_bag_size=<std_bag_size> --o=<output_dir> [--seed=<seed>]

Options:
  --model=<model_name>               Name of model to use for experiment
  --nbags=<nbags>                    Number of bags to generate
  --mean_bag_size=<mean_bag_size>    Mean size of sampled bags
  --std_bag_size=<std_bag_size>      Size standard deviation of sampled bags
  --o=<output_dir>                   Output directory
  --seed=<seed>                      Random seed
"""
import os
import logging
from docopt import docopt
import matplotlib.pyplot as plt
import generation as gen
import visualization as vis
from models import build_model, train_model, predict


N_SAMPLES_GROUNDTRUTH = 10000


def main(args):
    # Get model_name
    model_name = args['--model']
    output_dir = args['--o']

    # Generate bagged swiss roll dataset
    bags_sizes, individuals, bags_values, aggregate_targets, X_gt, t_gt = make_dataset(args)
    _ = vis.plot_dataset(individuals, X_gt, t_gt, aggregate_targets, bags_sizes)
    plt.savefig(os.path.join(output_dir, 'dataset.png'))
    plt.close()
    logging.info("Generated bag swiss roll dataset")

    # Create model
    model = build_model(name=model_name,
                        individuals=individuals,
                        bags_values=bags_values,
                        aggregate_targets=aggregate_targets,
                        bags_sizes=bags_sizes)
    logging.info(f"Initialized model {model}")

    # Fit hyperparameters
    logging.info("Fitting model hyperparameters")
    train_model(name=model_name, model=model)

    # Compute individuals predictive posterior and plot prediction
    individuals_posterior = predict(name=model_name, model=model, individuals=X_gt)
    _ = vis.plot_grountruth_prediction(individuals_posterior, X_gt, t_gt)
    plt.savefig(os.path.join(output_dir, 'prediction.png'))
    plt.close()


def make_dataset(args):
    """Generates bagged swiss-roll dataset

    Args:
        args (dict): input arguments

    Returns:
        type: list[int], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor

    """
    # Set aside random seed arg
    seed = int(args['--seed'])

    # Sample bags sizes
    bags_sizes = gen.sample_bags_sizes(mean_bag_size=int(args['--mean_bag_size']),
                                       std_bag_size=int(args['--std_bag_size']),
                                       n_bags=int(args['--nbags']),
                                       seed=seed)
    n_samples = sum(bags_sizes)

    # Generate groundtruth and uniform swiss rolls
    X_gt, t_gt = gen.make_swiss_roll(n_samples=N_SAMPLES_GROUNDTRUTH,
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


if __name__ == "__main__":
    # Read input args
    args = docopt(__doc__)

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logging.info(f'Arguments: {args}')

    # Run session
    main(args)
