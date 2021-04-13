import os
import yaml
import torch
import gpytorch
from sklearn.cluster import KMeans
from progress.bar import Bar
from models import VariationalCMEProcess, CMEProcessLikelihood, MODELS, TRAINERS, PREDICTERS
from core.metrics import compute_metrics, compute_chunked_nll


@MODELS.register('variational_cme_process')
def build_swiss_roll_variational_cme_process(n_inducing_points, lbda,
                                             individuals, bags_values, aggregate_targets,
                                             bags_sizes, use_individuals_noise, seed, **kwargs):
    """Hard-coded initialization of Variational CME Process module used for swiss roll experiment

    Args:
        n_inducing_points (int)
        lbda (float)
        individuals (torch.Tensor)
        bags_values (torch.Tensor)
        aggregate_targets (torch.Tensor)
        bags_sizes (list[int])
        use_individuals_noise (bool)
        seed (int)

    Returns:
        type: VariationalCMEProcess

    """
    # Inverse softplus utility for gpytorch lengthscale intialization
    inv_softplus = lambda x, n: torch.log(torch.exp(x * torch.ones(n)) - 1)

    # Define mean and covariance modules
    individuals_mean = gpytorch.means.ZeroMean()

    # Define individuals kernel
    base_individuals_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=3)
    base_individuals_kernel.initialize(raw_lengthscale=inv_softplus(x=1, n=3))
    individuals_kernel = gpytorch.kernels.ScaleKernel(base_individuals_kernel)

    # Define bags kernels
    base_bag_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=3)
    base_bag_kernel.initialize(raw_lengthscale=inv_softplus(x=1, n=3))
    bag_kernel = gpytorch.kernels.ScaleKernel(base_bag_kernel)

    # Initialize inducing points with kmeans
    kmeans = KMeans(n_clusters=n_inducing_points, init='k-means++', random_state=seed)
    kmeans.fit(individuals)
    inducing_points = torch.from_numpy(kmeans.cluster_centers_).float()

    # Define model
    model = VariationalCMEProcess(individuals_mean=individuals_mean,
                                  individuals_kernel=individuals_kernel,
                                  bag_kernel=bag_kernel,
                                  inducing_points=inducing_points,
                                  lbda=lbda,
                                  use_individuals_noise=use_individuals_noise)
    return model


@TRAINERS.register('variational_cme_process')
def train_swiss_roll_variational_cme_process(model, individuals, bags_values, aggregate_targets, bags_sizes,
                                             use_individuals_noise, lr, n_epochs, beta,
                                             groundtruth_individuals, groundtruth_targets, chunk_size, dump_dir, **kwargs):
    """Hard-coded training script of Variational CME Process for swiss roll experiment

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
    bags_values = bags_values.to(device)
    aggregate_targets = aggregate_targets.to(device)
    groundtruth_individuals = groundtruth_individuals.to(device)
    groundtruth_targets = groundtruth_targets.to(device)

    # Define variational CME process likelihood
    likelihood = CMEProcessLikelihood(use_individuals_noise=use_individuals_noise)

    # Set model in training mode
    model = model.train().to(device)
    likelihood = likelihood.train().to(device)

    # Define optimizer and elbo module
    parameters = list(model.parameters()) + list(likelihood.parameters())
    optimizer = torch.optim.Adam(params=parameters, lr=lr)
    elbo = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(aggregate_targets), beta=beta)

    # Extend bags tensor to match individuals size
    extended_bags_values = torch.cat([x.unsqueeze(0).repeat(bag_size, 1) for (x, bag_size) in zip(bags_values, bags_sizes)])

    # Initialize progress bar
    bar = Bar("Epoch", max=n_epochs)

    # Metrics record
    metrics = dict()

    for epoch in range(n_epochs):
        # Zero-out remaining gradients
        optimizer.zero_grad()

        # Compute q(f)
        q = model(individuals)

        # Compute tensors needed for ELBO computation
        elbo_kwargs = model.get_elbo_computation_parameters(bags_values=bags_values,
                                                            extended_bags_values=extended_bags_values)

        # Compute negative ELBO loss
        loss = -elbo(variational_dist_f=q,
                     target=aggregate_targets,
                     **elbo_kwargs)

        # Take gradient step
        loss.backward()
        optimizer.step()

        # Update progress bar
        bar.suffix = f"ELBO {-loss.item()}"
        bar.next()

        # Compute epoch metrics and dump
        epoch_metrics = compute_epoch_metrics(model=model,
                                              likelihood=likelihood,
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


def compute_epoch_metrics(model, likelihood, groundtruth_individuals, groundtruth_targets, chunk_size):
    # Set model in evaluation mode
    model.eval()

    # Compute individuals posterior on groundtruth distorted swiss roll
    individuals_posterior = predict_swiss_roll_variational_cme_process(model=model,
                                                                       individuals=groundtruth_individuals)
    # Compute MSE, MAE, MB
    epoch_metrics = compute_metrics(individuals_posterior=individuals_posterior, groundtruth_targets=groundtruth_targets)

    # Compute chunked approximation of NLL
    nll = compute_chunked_nll(groundtruth_individuals=groundtruth_individuals, groundtruth_targets=groundtruth_targets,
                              chunk_size=chunk_size, model=model, predict=predict_swiss_roll_variational_cme_process)
    epoch_metrics.update({'nll': nll})

    # Record model hyperparameters
    k_lengthscales = model.individuals_kernel.base_kernel.lengthscale.detach()[0].tolist()
    l_lengthscales = model.bag_kernel.base_kernel.lengthscale.detach()[0].tolist()
    epoch_metrics.update({'aggregate_noise': likelihood.noise.detach().item(),
                          'k_outputscale': model.individuals_kernel.outputscale.detach().item(),
                          'k_lengthscale_x': k_lengthscales[0],
                          'k_lengthscale_y': k_lengthscales[1],
                          'k_lengthscale_z': k_lengthscales[2],
                          'l_outputscale': model.bag_kernel.outputscale.detach().item(),
                          'l_lengthscale_x': l_lengthscales[0],
                          'l_lengthscale_y': l_lengthscales[1],
                          'l_lengthscale_z': l_lengthscales[2]})
    if model.noise_kernel:
        epoch_metrics.update({'indiv_noise': model.noise_kernel.outputscale.detach().item()})

    # Set model in train mode
    model.train()
    return epoch_metrics


@PREDICTERS.register('variational_cme_process')
def predict_swiss_roll_variational_cme_process(model, individuals, **kwargs):
    """Hard-coded prediciton of individuals posterior for Variational CME Process on
    swiss roll experiment

    Args:
        model (VariationalCMEProcess): in evaluation mode
        individuals (torch.Tensor)

    Returns:
        type: gpytorch.distributions.MultivariateNormal

    """
    # Compute predictive posterior on individuals
    with torch.no_grad():
        individuals_posterior = model(individuals)
    return individuals_posterior
