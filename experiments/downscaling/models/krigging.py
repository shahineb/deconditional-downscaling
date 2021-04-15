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
    base_spatial_kernel = RFFKernel(which='laplace', num_samples=500, ard_num_dims=3, active_dims=[0, 1, 2])
    base_spatial_kernel.initialize(raw_lengthscale=inv_softplus(x=1, n=3))

    base_feat_kernel = RFFKernel(which='laplace', num_samples=500, ard_num_dims=3, active_dims=[3, 4, 5])
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
                                           groundtruth_field, target_field, plot, **kwargs):
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

    # Define stochastic batch iterator
    def batch_iterator(batch_size):
        rdm_indices = torch.randperm(len(targets_blocks)).to(device)
        for idx in rdm_indices.split(batch_size):
            y = bags_blocks[idx]
            z = targets_blocks[idx]
            yield y, z

    # Define variational CME process likelihood
    likelihood = gpytorch.likelihood.GaussianLikelihood()

    # Set model in training mode
    model = model.train().to(device)
    likelihood = likelihood.train().to(device)

    # Define optimizer and elbo module
    parameters = list(model.parameters()) + list(likelihood.parameters())
    optimizer = torch.optim.Adam(params=parameters, lr=lr)
    elbo = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(targets_blocks), beta=beta)
    if seed:
        torch.random.manual_seed(seed)

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

            batch_bar.suffix = f"ELBO {-loss.item()}"
            batch_bar.next()

        # Compute posterior distribution at current epoch and store logs
        individuals_posterior = predict_downscaling_variational_krigging(model=model,
                                                                         covariates_grid=covariates_grid,
                                                                         mean_shift=mean_shift,
                                                                         std_scale=std_scale)
        epoch_logs = get_epoch_logs(model, likelihood, individuals_posterior, groundtruth_field)
        logs[epoch + 1] = epoch_logs
        with open(os.path.join(dump_dir, 'running_logs.yaml'), 'w') as f:
            yaml.dump({'epoch': logs}, f)

        # Dump plot of posterior prediction at current epoch
        if plot:
            _ = plot_downscaling_prediction(individuals_posterior, groundtruth_field, target_field)
            plt.savefig(os.path.join(dump_dir, f'png/epoch_{epoch}.png'))
            plt.close()
        epoch_bar.next()
        epoch_bar.finish()

    # Save model training state
    state = {'epoch': n_epochs,
             'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict()}
    torch.save(state, os.path.join(dump_dir, 'state.pt'))


def get_epoch_logs(model, likelihood, individuals_posterior, groundtruth_field):
    # Compute MSE, MAE, MB, Pearson Corr., SSIM
    epoch_logs = compute_metrics(individuals_posterior, groundtruth_field)

    # Record model hyperparameters
    k_spatial_kernel, k_feat_kernel = model.individuals_kernel.kernels
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
    mean_posterior = mean_shift + std_scale * individuals_posterior.mean
    lazy_covariance_posterior = (std_scale**2) * individuals_posterior.lazy_covariance_matrix
    output = gpytorch.distributions.MultivariateNormal(mean=mean_posterior,
                                                       covariance_matrix=lazy_covariance_posterior)

    # Set model back to training mode
    model.train()
    return output


# @PREDICTERS.register('krigging')
# def predict_downscaling_variational_krigging(model, covariates_grid, step_size, target_field, **kwargs):
#     """Hard-coded prediciton of individuals posterior for Variational CME Process on
#     downscaling experiment
#
#         Temporary not predicting covariance – need RFF inference first
#
#     Args:
#         model (ExactCMEProcess)
#         covariates_grid (torch.Tensor)
#         step_size (int)
#         target_field (torch.Tensor)
#
#     Returns:
#         type: gpytorch.distributions.MultivariateNormal
#
#     """
#     # Set model in evaluation mode
#     model.eval()
#
#     # Compute predictive posterior on individuals
#     bar = Bar("Predicting", max=covariates_grid.size(0) // step_size)
#     row_wise_pred = []
#     for i in range(0, covariates_grid.size(0), step_size):
#         col_wise_pred = []
#         for j in range(0, covariates_grid.size(1), step_size):
#             x_test = covariates_grid[i:i + step_size, j:j + step_size]
#             block_size = x_test.shape[:-1]
#             with torch.no_grad():
#                 individuals_posterior = model(x_test.reshape(-1, covariates_grid.size(-1)))
#             output = individuals_posterior.mean.reshape(*block_size)
#             col_wise_pred.append(output)
#         row_tensor = torch.cat(col_wise_pred, dim=1)
#         row_wise_pred.append(row_tensor)
#         bar.next()
#
#     # Encapsulate as MultivariateNormal
#     mean_posterior = torch.cat(row_wise_pred, dim=0).flatten() * target_field.values.std() + target_field.values.mean()
#     mean_posterior = mean_posterior.cpu()
#     dummy_covariance = gpytorch.lazy.DiagLazyTensor(diag=torch.ones_like(mean_posterior))
#     individuals_posterior = gpytorch.distributions.MultivariateNormal(mean=mean_posterior,
#                                                                       covariance_matrix=dummy_covariance)
#
#     # Set model back to training mode
#     model.train()
#     return individuals_posterior
