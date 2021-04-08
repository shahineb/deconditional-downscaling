import os
import yaml
import torch
import gpytorch
from progress.bar import Bar
from models import VariationalGP, VBaggGaussianLikelihood, MODELS, TRAINERS, PREDICTERS
from core.metrics import compute_metrics


@MODELS.register('vbagg')
def build_swiss_roll_vbagg_model(individuals, n_inducing_points, seed, **kwargs):
    """Hard-coded initialization of Vbagg model used for swiss roll experiment

    Args:
        individuals (torch.Tensor)
        n_inducing_points (int)
        seed (int)

    Returns:
        type: VariationalGP

    """
    # Inverse softplus utility for gpytorch lengthscale intialization
    inv_softplus = lambda x, n: torch.log(torch.exp(x * torch.ones(n)) - 1)

    # Define mean and covariance modules
    individuals_mean = gpytorch.means.ZeroMean()

    # Define individuals kernel
    base_individuals_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=3)
    base_individuals_kernel.initialize(raw_lengthscale=inv_softplus(x=1, n=3))
    individuals_kernel = gpytorch.kernels.ScaleKernel(base_individuals_kernel)

    # Select inducing points
    if seed:
        torch.random.manual_seed(seed)
    rdm_idx = torch.randperm(len(individuals))[:n_inducing_points]
    inducing_points = individuals[rdm_idx]

    # Define model
    model = VariationalGP(inducing_points=inducing_points,
                          mean_module=individuals_mean,
                          covar_module=individuals_kernel)
    return model


@TRAINERS.register('vbagg')
def train_swiss_roll_vbagg_model(model, individuals, aggregate_targets, bags_sizes,
                                 groundtruth_individuals, groundtruth_targets, lr, n_epochs, beta, dump_dir, **kwargs):
    """Hard-coded training script of Vbagg model for swiss roll experiment

    Args:
        model (VariationalGP)
        individuals (torch.Tensor)
        aggregate_targets (torch.Tensor)
        bags_sizes (list[int])
        lr (float)
        n_epochs (int)
        beta (float)

    """
    # Transfer on device
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    individuals = individuals.to(device)
    aggregate_targets = aggregate_targets.to(device)
    groundtruth_individuals = groundtruth_individuals.to(device)

    # Define VBAGG likelihood
    likelihood = VBaggGaussianLikelihood()

    # Set model in training mode
    model = model.train().to(device)
    likelihood = likelihood.train().to(device)

    # Define optimizer and elbo module
    parameters = list(model.parameters()) + list(likelihood.parameters())
    optimizer = torch.optim.Adam(params=parameters, lr=lr)
    elbo = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(aggregate_targets), beta=beta)

    # Initialize progress bar
    bar = Bar("Epoch", max=n_epochs)

    # Metrics record
    metrics = dict()

    for epoch in range(n_epochs):
        # Zero-out remaining gradients
        optimizer.zero_grad()

        # Compute q(f)
        q = model(individuals)

        # Compute negative ELBO loss
        loss = -elbo(variational_dist_f=q,
                     target=aggregate_targets,
                     bags_sizes=bags_sizes)

        # Take gradient step
        loss.backward()
        optimizer.step()

        bar.suffix = f"ELBO {-loss.item()}"
        bar.next()

        # Compute posterior distribution at current epoch and store metrics
        individuals_posterior = predict_swiss_roll_vbagg_model(model=model,
                                                               individuals=groundtruth_individuals)
        epoch_metrics = compute_metrics(individuals_posterior, groundtruth_targets)
        metrics[epoch + 1] = epoch_metrics
        with open(os.path.join(dump_dir, 'running_metrics.yaml'), 'w') as f:
            yaml.dump({'epoch': metrics}, f)

    # Save model training state
    state = {'epoch': n_epochs,
             'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict()}
    torch.save(state, os.path.join(dump_dir, 'state.pt'))


@PREDICTERS.register('vbagg')
def predict_swiss_roll_vbagg_model(model, individuals, **kwargs):
    """Hard-coded prediciton of individuals posterior for vbagg model on
    swiss roll experiment

    Args:
        model (VariationalGP)
        individuals (torch.Tensor)

    Returns:
        type: gpytorch.distributions.MultivariateNormal

    """
    # Set model in evaluation mode
    model.eval()

    # Compute predictive posterior on individuals
    with torch.no_grad():
        individuals_posterior = model(individuals)

    # Set back to trainind mode
    model.train()
    return individuals_posterior
