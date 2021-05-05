import os
import sys
import numpy as np
import torch

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../..")
sys.path.append(base_dir)

from src.utils import setseed


@setseed('numpy')
def sample_bags_sizes(mean_bag_size, std_bag_size, n_bags, seed=None):
    """Draws random list of bags sizes from negative binomial distribution
        with mean and std specified by given mean bag size and std of bags sizes

    Args:
        mean_bag_size (int): mean of negative binomial
        std_bag_size (int): std of negative binomial
        n_bags (int): number of samples to draw
        seed (int): random seed

    Returns:
        type: list[int]

    """
    # Compute probability of success and number of success parameters
    p = mean_bag_size / (std_bag_size**2)
    n = p * mean_bag_size / (1 - p)

    # Draw list of bags sizes
    bags_sizes = np.random.negative_binomial(n, p, size=n_bags).tolist()
    return bags_sizes


@setseed('torch')
def make_swiss_roll(n_samples, groundtruth=False, standardize=False, seed=None):
    """Draws random tensors of samples location on a swiss roll manifold following
        `sklearn.datasets.make_swiss_roll` implementation.

    Default behavior samples all dimensions uniformly. Generation of groundtruth
    samples used to compute aggregate observation values applies height-wise distortion
    of samples distribution

    Args:
        n_samples (int): number of samples
        groundtruth (bool): if True, applies height-wise distortion of distribution
        standardize (bool): if True, standardize features
        seed (int): random seed

    Returns:
        type: torch.Tensor, torch.Tensor

    """
    # Draw random samples uniformly in [0,1]
    samples = torch.rand(n_samples)

    # Compute position on manifold of each sample
    t = 1.5 * np.pi * (1 + 2 * samples)

    # Compute 3D coordinate of each manifold position
    x = t * torch.cos(t)
    z = t * torch.sin(t)
    y = 21 * torch.rand_like(samples)

    # Height-wise distortion of distribution for groundtruth roll
    if groundtruth:
        y.mul_(torch.sin(0.25 * np.pi * z))

    # Stack 3D coordinates together
    X = torch.stack([x, y, z], dim=-1).float()
    t = t.float()

    # Sort tensors by ascending height
    sorted_idx = torch.argsort(X[:, -1])
    X, t = X[sorted_idx], t[sorted_idx]

    # Standardize coordinate-wise
    if standardize:
        X = (X - X.mean(dim=0)) / X.std(dim=0)
        t = (t - t.mean()) / t.std()
    return X, t


def compose_bags_dataset(X, X_gt, t_gt, n_bags, noise):
    """Composes bagged dataset by splitting into bags along height and aggregating
        labels in distorted swiss roll

    Args:
        X (torch.Tensor): (n_samples, 3) uniform swiss roll tensor
        X_gt (torch.Tensor): (n_samples, 3) distorted swiss roll tensor
        t_gt (torch.Tensor): (n_samples,) distorted swiss roll labels
        n_bags (int): Number of bags
        noise (float): white noise variance on bags aggregate targets

    Returns:
        type: list[torch.Tensor], torch.Tensor, torch.Tensor

    """
    # Define splitting heights levels for bags
    lowest = X_gt[:, -1].min() - np.finfo(np.float16).eps
    highest = X_gt[:, -1].max() + np.finfo(np.float16).eps
    height_levels = torch.linspace(lowest, highest, n_bags + 1)
    bags_heights = list(zip(height_levels[:-1], height_levels[1:]))

    # Make bags mask for both swiss rolls based on heights
    bags_masks = [(X[:, -1] >= hmin) & (X[:, -1] < hmax) for (hmin, hmax) in bags_heights]
    bags_masks_gt = [(X_gt[:, -1] >= hmin) & (X_gt[:, -1] < hmax) for (hmin, hmax) in bags_heights]

    # Split uniform swiss roll into bags
    x = torch.cat([X[mask] for mask in bags_masks])
    bags_sizes = [mask.sum().item() for mask in bags_masks]

    # Compute average bag heights
    y = torch.tensor([np.mean(x) for x in bags_heights])

    # Compute noisy aggregate targets from distorted swiss roll
    z = torch.stack([t_gt[mask].mean() for mask in bags_masks_gt])
    z.add_(noise * torch.randn_like(z))
    return x, bags_sizes, y, z


def aggregate_bags(X, bags_sizes):
    """Splits 3D coordinate tensor into subtensors for each bag and takes
        mean value of their coordinate as bag value

    Args:
        X (torch.Tensor): (n_samples, 3) tensor
        bags_sizes (list[int]): sizes of bags s.t. Σbag_sizes = n_samples

    Returns:
        type: torch.Tensor, list[tuple[float]]

    """
    assert sum(bags_sizes) == X.size(0), "Mismatch between roll samples size and bags size"

    # Compute bag values by taking mean coordinate of each bag
    X_by_bag = X.split(bags_sizes)
    bags_values = torch.stack([x.mean(dim=0) for x in X_by_bag])

    # Compute minimum and maximum height of each bag
    bags_heights = [[x[:, -1].min().item(), x[:, -1].max().item()] for x in X_by_bag]
    return bags_values, bags_heights


@setseed('torch')
def aggregate_targets(X, t, bags_heights, individuals_noise, aggregate_noise, seed=None):
    """Splits targets by bags height chunks and averages them

    Args:
        X (torch.Tensor): (n_samples, 3) tensor
        t (torch.Tensor): (n_samples,)
        bags_heights (list[tuple[float]]): [(zmin, zmax)] of each bag
        individuals_noise (float): noise variance to add on individuals targets
        aggregate_noise (float): noise variance to add on aggregate targets

    Returns:
        type: Description of returned object.

    """
    # Add random noise on individuals targets
    noisy_t = t.add(individuals_noise * torch.randn_like(t))

    # Adjust lowest/highest heights to match samples in case different tensor
    bags_heights[0][0] = X[:, -1].min().item()
    bags_heights[-1][1] = X[:, -1].max().item()

    # Compute aggregate target values by averaging over height chunks
    bags_masks = [(X[:, -1] >= zmin) & (X[:, -1] < zmax) for (zmin, zmax) in bags_heights]
    aggregate_t = torch.stack([noisy_t[mask].mean() for mask in bags_masks])

    # Add random noise on aggregate observations
    aggregate_t.add_(aggregate_noise * torch.randn_like(aggregate_t))
    return aggregate_t
