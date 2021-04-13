import os
import yaml
import torch
import gpytorch
from progress.bar import Bar
from models import ExactGP, MODELS, TRAINERS, PREDICTERS
from core.metrics import compute_metrics, compute_chunked_nll


@MODELS.register('gp_regression')
def build_swiss_roll_gp_regressor(bags_values, aggregate_targets, **kwargs):
    """Hard-coded initialization of ExactGP module used for swiss roll experiment

    Args:
        bags_values (torch.Tensor)
        aggregate_targets (torch.Tensor)

    Returns:
        type: ExactGP

    """
    # Inverse softplus utility for gpytorch lengthscale intialization
    inv_softplus = lambda x, n: torch.log(torch.exp(x * torch.ones(n)) - 1)

    # Define mean and covariance modules
    mean_module = gpytorch.means.ZeroMean()

    # Define individuals kernel
    base_individuals_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=3)
    base_individuals_kernel.initialize(raw_lengthscale=inv_softplus(x=1, n=3))
    covar_module = gpytorch.kernels.ScaleKernel(base_individuals_kernel)

    # Define model
    model = ExactGP(mean_module=mean_module,
                    covar_module=covar_module,
                    train_x=bags_values,
                    train_y=aggregate_targets,
                    likelihood=gpytorch.likelihoods.GaussianLikelihood())
    return model


@TRAINERS.register('gp_regression')
def train_swiss_roll_gp_regressor(model, bags_values, aggregate_targets, lr, n_epochs,
                                  groundtruth_individuals, groundtruth_targets,
                                  chunk_size, dump_dir, **kwargs):
    """Hard-coded training script of Exact GP for swiss roll experiment

    Args:
        model (ExactGP)
        lr (float)
        n_epochs (int)
        groundtruth_individuals (torch.Tensor)
        groundtruth_targets (torch.Tensor)
        dump_dir (str)

    """
    # Move tensors to device
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    bags_values = bags_values.to(device)
    aggregate_targets = aggregate_targets.to(device)
    groundtruth_individuals = groundtruth_individuals.to(device)
    groundtruth_targets = groundtruth_targets.to(device)

    # Set model in training mode
    model = model.train().to(device)

    # Define optimizer and exact loglikelihood module
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    # Initialize progress bar
    bar = Bar("Epoch", max=n_epochs)

    # Metrics record
    metrics = dict()

    for epoch in range(n_epochs):
        # Zero-out remaining gradients
        optimizer.zero_grad()

        # Compute marginal distribution p(.|x,y)
        output = model(bags_values)

        # Evaluate -logp(z|x, y) on aggregate observations z
        loss = -mll(output, aggregate_targets)

        # Take gradient step
        loss.backward()
        optimizer.step()

        # Update progress bar
        bar.suffix = f"NLL {loss.item()}"
        bar.next()

        # Compute posterior distribution at current epoch and store metrics
        epoch_metrics = compute_epoch_metrics(model=model,
                                              groundtruth_individuals=groundtruth_individuals,
                                              groundtruth_targets=groundtruth_targets,
                                              chunk_size=chunk_size)
        metrics[epoch + 1] = epoch_metrics
        with open(os.path.join(dump_dir, 'running_metrics.yaml'), 'w') as f:
            yaml.dump({'epoch': metrics}, f)

    # Save model training state
    state = {'epoch': n_epochs,
             'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict()}
    torch.save(state, os.path.join(dump_dir, 'state.pt'))


def compute_epoch_metrics(model, groundtruth_individuals, groundtruth_targets, chunk_size):
    # Set model in evaluation mode
    model.eval()

    # Compute individuals posterior on groundtruth distorted swiss roll
    individuals_posterior = predict_swiss_roll_gp_regressor(model=model,
                                                            individuals=groundtruth_individuals)
    # Compute MSE, MAE, MB
    epoch_metrics = compute_metrics(individuals_posterior=individuals_posterior, groundtruth_targets=groundtruth_targets)

    # Compute chunked approximation of NLL
    nll = compute_chunked_nll(groundtruth_individuals=groundtruth_individuals, groundtruth_targets=groundtruth_targets,
                              chunk_size=chunk_size, model=model, predict=predict_swiss_roll_gp_regressor)
    epoch_metrics.update({'nll': nll})

    # Record model hyperparameters
    lengthscales = model.covar_module.base_kernel.lengthscale.detach()[0].tolist()
    epoch_metrics.update({'aggregate_noise': model.likelihood.noise.detach().item(),
                          'outputscale': model.covar_module.outputscale.detach().item(),
                          'lengthscale_x': lengthscales[0],
                          'lengthscale_y': lengthscales[1],
                          'lengthscale_z': lengthscales[2]})

    # Clear model cache from prediction strategy
    model._clear_cache()

    # Set model in train mode
    model.train()
    return epoch_metrics


@PREDICTERS.register('gp_regression')
def predict_swiss_roll_gp_regressor(model, individuals, **kwargs):
    """Hard-coded prediciton of individuals posterior for ExactGP on
    swiss roll experiment

    Args:
        model (ExactGP): in evaluation mode
        individuals (torch.Tensor)

    Returns:
        type: gpytorch.distributions.MultivariateNormal

    """
    # Compute predictive posterior on individuals
    with torch.no_grad():
        individuals_posterior = model(individuals)
    return individuals_posterior
