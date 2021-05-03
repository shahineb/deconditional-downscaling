import os
import yaml
import logging
import torch
import gpytorch
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from progress.bar import Bar
from models import VariationalGP, RFFKernel, MODELS, TRAINERS, PREDICTERS
from core.visualization import plot_downscaling_prediction
from core.metrics import compute_metrics


@MODELS.register('krigging')
def build_downscaling_variational_krigging(bags_blocks, n_inducing_points, seed, **kwargs):
    """Hard-coded initialization of ExactGP module used for downscaling experiment

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
    base_spatial_kernel = RFFKernel(nu=1.5, num_samples=1000, ard_num_dims=3, active_dims=[0, 1, 2])
    base_spatial_kernel.initialize(raw_lengthscale=inv_softplus(x=1, n=3))

    base_feat_kernel = gpytorch.kernels.RFFKernel(num_samples=1000, ard_num_dims=3, active_dims=[3, 4, 5])
    base_feat_kernel.initialize(raw_lengthscale=inv_softplus(x=1, n=3))

    spatial_kernel = gpytorch.kernels.ScaleKernel(base_spatial_kernel)
    feat_kernel = gpytorch.kernels.ScaleKernel(base_feat_kernel)
    covar_module = spatial_kernel + feat_kernel

    # Initialize inducing points with kmeans
    if seed:
        torch.random.manual_seed(seed)
    rdm_idx = torch.randperm(bags_blocks.size(0))[:n_inducing_points]
    inducing_points = bags_blocks[rdm_idx].float()
    # kmeans = KMeans(n_clusters=n_inducing_points, init='k-means++', random_state=seed)
    # kmeans.fit(bags_blocks)
    # inducing_points = torch.from_numpy(kmeans.cluster_centers_).float()

    # Define model
    model = VariationalGP(inducing_points=inducing_points,
                          mean_module=mean_module,
                          covar_module=covar_module)
    return model


@TRAINERS.register('krigging')
def train_downscaling_variational_krigging(model, bags_blocks, targets_blocks,
                                           lr, n_epochs, batch_size, beta, seed, dump_dir, covariates_grid,
                                           groundtruth_field, target_field, missing_bags_fraction, plot, plot_every, **kwargs):
    """Hard-coded training script of Vbagg model for downscaling experiment

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
    covariates_grid = covariates_grid.to(device)
    bags_blocks = bags_blocks.to(device)
    targets_blocks = targets_blocks.to(device)

    # Drop some bags
    if seed:
        torch.random.manual_seed(seed)
    n_drop = int(missing_bags_fraction * len(targets_blocks))
    drop_idx = torch.randperm(len(targets_blocks)).to(device)[:n_drop]
    bags_blocks = bags_blocks[~drop_idx]
    targets_blocks = targets_blocks[~drop_idx]

    # Define stochastic batch iterator
    def batch_iterator(batch_size):
        rdm_indices = torch.randperm(len(targets_blocks)).to(device)
        for idx in rdm_indices.split(batch_size):
            y = bags_blocks[idx]
            z = targets_blocks[idx]
            yield y, z

    # Define Gaussian likelihood
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    # Set model in training mode
    model = model.train().to(device)
    likelihood = likelihood.train().to(device)

    # Define optimizer and elbo module
    parameters = list(model.parameters()) + list(likelihood.parameters())
    optimizer = torch.optim.Adam(params=parameters, lr=lr)
    elbo = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(targets_blocks), beta=beta)

    # Compute unnormalization mean shift and scaling for prediction
    mean_shift = target_field.values.mean()
    std_scale = target_field.values.std()

    # Initialize progress bar
    epoch_bar = Bar("Epoch", max=n_epochs)
    epoch_bar.finish()

    # Metrics record
    logs = dict()

    for epoch in range(n_epochs):

        batch_bar = Bar("Batch", max=len(targets_blocks) // batch_size)
        epoch_loss = 0

        for y, z in batch_iterator(batch_size):
            # Zero-out remaining gradients
            optimizer.zero_grad()

            # Compute q(f)
            q = model(y)

            # Compute negative ELBO loss
            loss = -elbo(variational_dist_f=q, target=z)

            # Take gradient step
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_bar.suffix = f"Running ELBO {-loss.item()}"
            batch_bar.next()

        # Compute posterior distribution at current epoch and store logs
        individuals_posterior = predict_downscaling_variational_krigging(model=model,
                                                                         covariates_grid=covariates_grid,
                                                                         mean_shift=mean_shift,
                                                                         std_scale=std_scale)
        epoch_logs = get_epoch_logs(model, likelihood, individuals_posterior, groundtruth_field)
        epoch_logs.update({'loss': epoch_loss / (len(targets_blocks) // batch_size)})
        logs[epoch + 1] = epoch_logs
        with open(os.path.join(dump_dir, 'running_logs.yaml'), 'w') as f:
            yaml.dump({'epoch': logs}, f)

        # Dump plot of posterior prediction at current epoch
        if plot and epoch % plot_every == 0:
            _ = plot_downscaling_prediction(individuals_posterior, groundtruth_field, target_field, drop_idx)
            plt.savefig(os.path.join(dump_dir, f'png/epoch_{epoch}.png'))
            plt.close()
        epoch_bar.next()
        epoch_bar.finish()

        # Empty cache if using GPU
        if torch.cuda.is_available():
            del individuals_posterior
            torch.cuda.empty_cache()

    # Save model training state
    state = {'epoch': n_epochs,
             'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict()}
    torch.save(state, os.path.join(dump_dir, 'state.pt'))


def get_epoch_logs(model, likelihood, individuals_posterior, groundtruth_field):
    # Compute MSE, MAE, MB, Pearson Corr., SSIM
    epoch_logs = compute_metrics(individuals_posterior, groundtruth_field)

    # Record model hyperparameters
    k_spatial_kernel, k_feat_kernel = model.covar_module.kernels
    k_spatial_lengthscales = k_spatial_kernel.base_kernel.lengthscale[0].detach().tolist()
    k_feat_lengthscales = k_feat_kernel.base_kernel.lengthscale[0].detach().tolist()

    epoch_logs.update({'aggregate_noise': likelihood.noise.detach().item(),
                       'k_spatial_outputscale': k_spatial_kernel.outputscale.detach().item(),
                       'k_lengthscale_lat': k_spatial_lengthscales[0],
                       'k_lengthscale_lon': k_spatial_lengthscales[1],
                       'k_lengthscale_alt': k_spatial_lengthscales[2],
                       'k_feat_outputscale': k_feat_kernel.outputscale.detach().item(),
                       'k_lengthscale_albisccp': k_feat_lengthscales[0],
                       'k_lengthscale_clt': k_feat_lengthscales[1],
                       'k_lengthscale_pctisccp': k_feat_lengthscales[2]})
    return epoch_logs


@PREDICTERS.register('krigging')
def predict_downscaling_variational_krigging(model, covariates_grid, mean_shift, std_scale, **kwargs):
    # Set model in evaluation mode
    model.eval()

    # Compute standardized posterior distribution on individuals
    with torch.no_grad():
        logging.info("\n Infering deconditioning posterior on HR pixels...")
        individuals_posterior = model(covariates_grid.view(-1, covariates_grid.size(-1)))

    # Rescale by mean and std from observed aggregate target field
    mean_posterior = mean_shift + std_scale * individuals_posterior.mean.cpu()
    lazy_covariance_posterior = (std_scale**2) * individuals_posterior.lazy_covariance_matrix.cpu()
    output = gpytorch.distributions.MultivariateNormal(mean=mean_posterior,
                                                       covariance_matrix=lazy_covariance_posterior)

    # Set model back to training mode
    model.train()
    return output
