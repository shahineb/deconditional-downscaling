import os
import yaml
import torch
import gpytorch
import matplotlib.pyplot as plt
from progress.bar import Bar
from models import VariationalCMEProcess, CMEProcessLikelihood, MODELS, TRAINERS, PREDICTERS
from core.visualization import plot_downscaling_prediction
from core.metrics import compute_metrics


@MODELS.register('variational_cme_process')
def build_downscaling_variational_cme_process(covariates_blocks, lbda, n_inducing_points, seed, **kwargs):
    """Hard-coded initialization of Variational CME Process module used for downscaling experiment

    Args:
        individuals (torch.Tensor)
        lbda (float)
        n_inducing_points (int)
        seed (int)

    Returns:
        type: VariationalCMEProcess

    """
    # Inverse softplus utility for gpytorch lengthscale intialization
    inv_softplus = lambda x, n: torch.log(torch.exp(x * torch.ones(n)) - 1)

    # Define mean and covariance modules
    individuals_mean = gpytorch.means.ZeroMean()

    # Define individuals kernel
    base_indiv_spatial_kernel = gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=3, active_dims=[0, 1, 2])
    base_indiv_spatial_kernel.initialize(raw_lengthscale=inv_softplus(x=1, n=3))

    base_indiv_feat_kernel = gpytorch.kernels.MaternKernel(nu=0.5, ard_num_dims=3, active_dims=[3, 4, 5])
    base_indiv_feat_kernel.initialize(raw_lengthscale=inv_softplus(x=1, n=3))

    individuals_spatial_kernel = gpytorch.kernels.ScaleKernel(base_indiv_spatial_kernel)
    individuals_feat_kernel = gpytorch.kernels.ScaleKernel(base_indiv_feat_kernel)
    individuals_kernel = individuals_spatial_kernel + individuals_feat_kernel

    # Define bags kernels
    base_bag_spatial_kernel = gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=3, active_dims=[0, 1, 2])
    base_bag_spatial_kernel.initialize(raw_lengthscale=inv_softplus(x=1, n=3))

    base_bag_feat_kernel = gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=3, active_dims=[3, 4, 5])
    base_bag_feat_kernel.initialize(raw_lengthscale=inv_softplus(x=1, n=3))

    bag_spatial_kernel = gpytorch.kernels.ScaleKernel(base_bag_spatial_kernel)
    bag_feat_kernel = gpytorch.kernels.ScaleKernel(base_bag_feat_kernel)
    bag_kernel = bag_spatial_kernel + bag_feat_kernel

    # Select inducing points
    if seed:
        torch.random.manual_seed(seed)
    rdm_idx = torch.randperm(len(covariates_blocks))[:n_inducing_points]
    inducing_points = covariates_blocks[rdm_idx].mean(dim=1).float()

    # Define model
    model = VariationalCMEProcess(individuals_mean=individuals_mean,
                                  individuals_kernel=individuals_kernel,
                                  bag_kernel=bag_kernel,
                                  inducing_points=inducing_points,
                                  lbda=lbda)
    return model


@TRAINERS.register('variational_cme_process')
def train_downscaling_variational_cme_process(model, covariates_blocks, bags_blocks, extended_bags, targets_blocks,
                                              lr, n_epochs, batch_size, beta, seed, dump_dir, covariates_grid,
                                              step_size, groundtruth_field, target_field, plot, **kwargs):
    """Hard-coded training script of Exact CME Process for downscaling experiment

    Args:
        model (VariationalGP)
        covariates_blocks (torch.Tensor)
        bags_blocks (torch.Tensor)
        extended_bags (torch.Tensor)
        targets_blocks (torch.Tensor)
        lr (float)
        n_epochs (int)
        beta (float)
        batch_size (int)
        seed (int)
        dump_dir (str)
        covariates_grid (torch.Tensor)
        step_size (int)
        groundtruth_field (xarray.core.dataarray.DataArray)
        target_field (torch.Tensor)
        plot (bool)

    """
    # Transfer on device
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    covariates_grid = covariates_grid.to(device)
    covariates_blocks = covariates_blocks.to(device)
    bags_blocks = bags_blocks.to(device)
    extended_y = extended_bags.to(device)
    targets_blocks = targets_blocks.to(device)

    # Define stochastic batch iterator
    def batch_iterator(batch_size):
        rdm_indices = torch.randperm(len(targets_blocks))
        n_dim_individuals = covariates_blocks.size(-1)
        n_dim_bags = bags_blocks.size(-1)
        for idx in rdm_indices.split(batch_size):
            x = covariates_blocks[idx].reshape(-1, n_dim_individuals)
            y = bags_blocks[idx]
            extended_y = extended_bags[idx].reshape(-1, n_dim_bags)
            z = targets_blocks[idx]
            yield x, y, extended_y, z

    # Define variational CME process likelihood
    likelihood = CMEProcessLikelihood()

    # Set model in training mode
    model.train()
    likelihood.train()
    model = model.to(device)
    likelihood = likelihood.to(device)

    # Define optimizer and elbo module
    parameters = list(model.parameters()) + list(likelihood.parameters())
    optimizer = torch.optim.Adam(params=parameters, lr=lr)
    elbo = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(targets_blocks), beta=beta)
    if seed:
        torch.random.manual_seed(seed)

    # Initialize progress bar
    epoch_bar = Bar("Epoch", max=n_epochs)
    epoch_bar.finish()

    # Metrics record
    metrics = dict()

    for epoch in range(n_epochs):

        batch_bar = Bar("Batch", max=len(targets_blocks) // batch_size)

        for x, y, extended_y, z in batch_iterator(batch_size):
            # Zero-out remaining gradients
            optimizer.zero_grad()

            # Compute q(f)
            q = model(x)

            # Compute tensors needed for ELBO computation
            root_inv_extended_bags_covar, bags_to_extended_bags_covar = model.get_elbo_computation_parameters(bags_values=y,
                                                                                                              extended_bags_values=extended_y)

            # Compute negative ELBO loss
            loss = -elbo(variational_dist_f=q,
                         target=z,
                         root_inv_extended_bags_covar=root_inv_extended_bags_covar,
                         bags_to_extended_bags_covar=bags_to_extended_bags_covar)

            # Take gradient step
            loss.backward()
            optimizer.step()

            batch_bar.suffix = f"ELBO {-loss.item()}"
            batch_bar.next()

        # Compute posterior distribution at current epoch and store metrics
        individuals_posterior = predict_downscaling_variational_cme_process(model=model,
                                                                            covariates_grid=covariates_grid,
                                                                            step_size=step_size,
                                                                            target_field=target_field)
        epoch_metrics = compute_metrics(individuals_posterior, groundtruth_field)
        metrics[epoch + 1] = epoch_metrics
        with open(os.path.join(dump_dir, 'running_metrics.yaml'), 'w') as f:
            yaml.dump({'epoch': metrics}, f)

        # Dump plot of posterior prediction at current epoch
        if plot:
            _ = plot_downscaling_prediction(individuals_posterior, groundtruth_field, target_field)
            plt.savefig(os.path.join(dump_dir, f'png/epoch_{epoch}.png'))
            plt.close()
        epoch_bar.next()
        epoch_bar.finish()


@PREDICTERS.register('variational_cme_process')
def predict_downscaling_variational_cme_process(model, covariates_grid, step_size, target_field, **kwargs):
    """Hard-coded prediciton of individuals posterior for Variational CME Process on
    downscaling experiment

        Temporary not predicting covariance – need RFF inference first

    Args:
        model (ExactCMEProcess)
        covariates_grid (torch.Tensor)
        step_size (int)
        target_field (torch.Tensor)

    Returns:
        type: gpytorch.distributions.MultivariateNormal

    """
    # Set model in evaluation mode
    model.eval()

    # Compute predictive posterior on individuals
    bar = Bar("Predicting", max=covariates_grid.size(0) // step_size)
    row_wise_pred = []
    for i in range(0, covariates_grid.size(0), step_size):
        col_wise_pred = []
        for j in range(0, covariates_grid.size(1), step_size):
            x_test = covariates_grid[i:i + step_size, j:j + step_size]
            block_size = x_test.shape[:-1]
            with torch.no_grad():
                individuals_posterior = model(x_test.reshape(-1, covariates_grid.size(-1)))
            output = individuals_posterior.mean.reshape(*block_size)
            col_wise_pred.append(output)
        row_tensor = torch.cat(col_wise_pred, dim=1)
        row_wise_pred.append(row_tensor)
        bar.next()

    # Encapsulate as MultivariateNormal
    mean_posterior = torch.cat(row_wise_pred, dim=0).flatten() * target_field.values.std() + target_field.values.mean()
    mean_posterior = mean_posterior.cpu()
    dummy_covariance = gpytorch.lazy.DiagLazyTensor(diag=torch.ones_like(mean_posterior))
    individuals_posterior = gpytorch.distributions.MultivariateNormal(mean=mean_posterior,
                                                                      covariance_matrix=dummy_covariance)

    # Set model back to training mode
    model.train()
    return individuals_posterior
