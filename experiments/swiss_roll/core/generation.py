import os
import sys
import numpy as np
import torch
import gpytorch
from progress.bar import Bar

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../..")
sys.path.append(base_dir)

from src.models import ExactGP
from src.utils import setseed


@setseed('torch')
def make_swiss_roll(n_samples, standardize=False, seed=None):
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


@setseed('torch')
def compose_bags_dataset(X, t, n_bags, noise, seed=None):
    """Composes bagged dataset by splitting into bags along height and aggregating
        labels in distorted swiss roll

    Args:
        X (torch.Tensor): (n_samples, 3) uniform swiss roll tensor
        X_gt (torch.Tensor): (n_samples, 3) distorted swiss roll tensor
        t_gt (torch.Tensor): (n_samples,) distorted swiss roll labels
        n_bags (int): Number of bags
        noise (float): white noise variance on bags aggregate targets

    Returns:
        type: torch.Tensor, torch.Tensor, list[torch.Tensor], torch.Tensor, torch.Tensor

    """
    # Define splitting heights levels for bags
    lowest = X[:, -1].min() - np.finfo(np.float16).eps
    highest = X[:, -1].max() + np.finfo(np.float16).eps
    height_levels = torch.linspace(lowest, highest, n_bags + 1)
    bags_heights = list(zip(height_levels[:-1], height_levels[1:]))

    # Make bags mask for both swiss rolls based on heights
    bags_masks = [(X[:, -1] >= hmin) & (X[:, -1] < hmax) for (hmin, hmax) in bags_heights]

    # Compute average bag heights
    y = torch.tensor([np.mean(x) for x in bags_heights])

    # Compute noisy aggregate targets from distorted swiss roll
    z = torch.stack([t[mask].mean() for mask in bags_masks])
    z.add_(noise * torch.randn_like(z))

    # Split uniform swiss roll into bags
    x = torch.cat([X[mask] for mask in bags_masks])
    bags_sizes = [mask.sum().item() for mask in bags_masks]

    # Extend bags tensor to match individuals size
    extended_y = torch.cat([bag_value.repeat(bag_size, 1) for (bag_size, bag_value) in zip(bags_sizes, y)]).squeeze()
    return x, extended_y, bags_sizes, y, z


@setseed('torch')
def unmatch_datasets(x, extended_y, bags_sizes, y, z, split, seed=None):
    """Short summary.

    Args:
        x (torch.Tensor): (n_samples, 3) tensor of individuals covariates
        extended_y (torch.Tensor): (n_samples,) tensor of replicated bag-level covariates
        bags_sizes (list[int]): list of bags sizes for above two tensors
        y (torch.Tensor): (n_bags,) tensor of bag-level covariates
        z (torch.Tensor): (n_bags,) tensor of bag aggregate targets
        split (float): fraction of dataset to use for D1
        seed (int): random seed

    Returns:
        type: torch.Tensor, torch.Tensor, list[torch.Tensor], torch.Tensor, torch.Tensor

    """
    # Sample random permutation of bags indices
    rdm_idx = torch.randperm(len(bags_sizes))

    # Split into 2 separate subsets
    N1 = int(split * len(bags_sizes))
    indices_1, indices_2 = rdm_idx[:N1], rdm_idx[N1:]

    # Subset x and extended_y to first subset of indices
    x_by_bag = x.split(bags_sizes)
    x = torch.cat([x_by_bag[idx] for idx in indices_1])

    extended_y_by_bag = extended_y.split(bags_sizes)
    extended_y = torch.cat([extended_y_by_bag[idx] for idx in indices_1]).squeeze()

    bags_sizes = [bags_sizes[idx] for idx in indices_1]

    # Subset y and z to second subset of indices
    y = y[indices_2]
    z = z[indices_2]
    return x, extended_y, bags_sizes, y, z


def make_mediating_gp_regressor(y, z, lr, n_epochs):
    """Fit standard GP regression model between bag-level covariates and
        bag aggregate targets

    Args:
        y (torch.Tensor): (n_bags,) tensor of bag-level covariates
        z (torch.Tensor): (n_bags,) tensor of bag aggregate targets

    Returns:
        type: ExactGP

    """
    # Inverse softplus utility for gpytorch lengthscale intialization
    inv_softplus = lambda x, n: torch.log(torch.exp(x * torch.ones(n)) - 1)

    # Define mean and covariance modules
    bag_mean = gpytorch.means.ZeroMean()

    # Define bags kernels
    base_bag_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=1)
    base_bag_kernel.initialize(raw_lengthscale=inv_softplus(x=1, n=1))
    bag_kernel = gpytorch.kernels.ScaleKernel(base_bag_kernel)

    # Define model
    model = ExactGP(mean_module=bag_mean,
                    covar_module=bag_kernel,
                    train_x=y,
                    train_y=z,
                    likelihood=gpytorch.likelihoods.GaussianLikelihood())

    # Set model in training mode
    model = model.train()

    # Define optimizer and exact loglikelihood module
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    # Initialize progress bar
    bar = Bar("Epoch", max=n_epochs)

    for epoch in range(n_epochs):
        # Zero-out remaining gradients
        optimizer.zero_grad()

        # Compute marginal distribution p(.|x,y)
        output = model(model.train_inputs[0])

        # Evaluate -logp(z|x, y) on aggregate observations z
        loss = -mll(output, model.train_targets)

        # Take gradient step
        loss.backward()
        optimizer.step()

        # Evaluate model
        model.eval()
        with torch.no_grad():
            posterior = model(model.train_inputs[0])
            mse = torch.pow(posterior.mean - model.train_targets, 2).mean()
        model.train()

        # Update progress bar
        bar.suffix = f"NLL {loss.item()} - MSE {mse.item()}"
        bar.next()
    return model.eval()
