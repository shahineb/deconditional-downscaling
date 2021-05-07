import os
import yaml
import torch
import gpytorch
from progress.bar import Bar
from models import BaggedGP, VBaggGaussianLikelihood, MODELS, TRAINERS, PREDICTERS
from core.metrics import compute_metrics, compute_chunked_nll


@MODELS.register('bagged_gp')
def build_swiss_roll_bagged_gp(individuals, aggregate_targets, bags_sizes, independent_bags, **kwargs):
    """Hard-coded initialization of Exact CME Process module used for swiss roll experiment

    Args:
        individuals (torch.Tensor)
        bags_values (torch.Tensor)
        aggregate_targets (torch.Tensor)
        bags_sizes (list[int])
        lbda (float)
        use_individuals_noise (bool)

    Returns:
        type: ExactCMEProcess

    """
    # Inverse softplus utility for gpytorch lengthscale intialization
    inv_softplus = lambda x, n: torch.log(torch.exp(x * torch.ones(n)) - 1)

    # Define mean and covariance modules
    individuals_mean = gpytorch.means.ZeroMean()

    # Define individuals kernel
    base_individuals_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=3)
    base_individuals_kernel.initialize(raw_lengthscale=inv_softplus(x=1, n=3))
    individuals_kernel = gpytorch.kernels.ScaleKernel(base_individuals_kernel)

    # Define model
    model = BaggedGP(individuals_mean=individuals_mean,
                     individuals_kernel=individuals_kernel,
                     train_individuals=individuals,
                     train_aggregate_targets=aggregate_targets,
                     bags_sizes=bags_sizes,
                     independent_bags=independent_bags,
                     likelihood=VBaggGaussianLikelihood())
    return model


@TRAINERS.register('bagged_gp')
def train_swiss_roll_bagged_gp(model, lr, n_epochs,
                               groundtruth_individuals, groundtruth_targets,
                               chunk_size, dump_dir, **kwargs):
    """Hard-coded training script of Exact CME Process for swiss roll experiment

    Args:
        model (ExactCMEProcess)
        lr (float)
        n_epochs (int)

    """
    # Move to device
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    groundtruth_targets = groundtruth_targets.to(device)

    # Set model in training mode
    model.train()
    model.likelihood.train()

    # Define optimizer and exact loglikelihood module
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    # Initialize progress bar
    bar = Bar("Epoch", max=n_epochs)

    # Logs record
    logs = dict()

    for epoch in range(n_epochs):
        # Zero-out remaining gradients
        optimizer.zero_grad()

        # Compute marginal distribution p(.|x)
        output = model(model.train_individuals, model.bags_sizes)

        # Evaluate -logp(z|x) on aggregate observations z
        loss = -mll(output, model.train_aggregate_targets, model.bags_sizes)

        # Take gradient step
        loss.backward()
        optimizer.step()

        # Update progress bar
        bar.suffix = f"NLL {loss.item()}"
        bar.next()

        # Compute epoch logs and dump
        epoch_logs = get_epoch_logs(model=model,
                                    groundtruth_individuals=groundtruth_individuals,
                                    groundtruth_targets=groundtruth_targets,
                                    chunk_size=chunk_size)
        epoch_logs.update(loss=loss.item())
        logs[epoch + 1] = epoch_logs
        with open(os.path.join(dump_dir, 'running_logs.yaml'), 'w') as f:
            yaml.dump({'epoch': logs}, f)

    # Save model training state
    state = {'epoch': n_epochs,
             'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict()}
    torch.save(state, os.path.join(dump_dir, 'state.pt'))


def get_epoch_logs(model, groundtruth_individuals, groundtruth_targets, chunk_size):
    # Compute individuals posterior on groundtruth distorted swiss roll
    individuals_posterior = predict_swiss_roll_bagged_gp(model=model, individuals=groundtruth_individuals)
    # Compute MSE, MAE, MB
    epoch_logs = compute_metrics(individuals_posterior=individuals_posterior, groundtruth_targets=groundtruth_targets)

    # Compute chunked approximation of NLL
    nll = compute_chunked_nll(groundtruth_individuals=groundtruth_individuals, groundtruth_targets=groundtruth_targets,
                              chunk_size=chunk_size, model=model, predict=predict_swiss_roll_bagged_gp)
    epoch_logs.update({'nll': nll})

    # Record model hyperparameters
    k_lengthscales = model.individuals_kernel.base_kernel.lengthscale.detach()[0].tolist()
    epoch_logs.update({'aggregate_noise': model.likelihood.noise.detach().item(),
                       'k_outputscale': model.individuals_kernel.outputscale.detach().item(),
                       'k_lengthscale_x': k_lengthscales[0],
                       'k_lengthscale_y': k_lengthscales[1],
                       'k_lengthscale_z': k_lengthscales[2]})

    # Clear model cache from prediction strategy
    model._clear_cache()
    return epoch_logs


@PREDICTERS.register('bagged_gp')
def predict_swiss_roll_bagged_gp(model, individuals, **kwargs):
    """Hard-coded prediciton of individuals posterior for Bagged GP model on
    swiss roll experiment

    Args:
        model (BaggedGP)
        individuals (torch.Tensor)

    Returns:
        type: gpytorch.distributions.MultivariateNormal

    """
    are_bags_independents = model.independent_bags
    model.independent_bags = False
    # Compute predictive posterior on individuals
    with torch.no_grad():
        individuals_posterior = model.predict(individuals)
    model.independent_bags = are_bags_independents
    return individuals_posterior