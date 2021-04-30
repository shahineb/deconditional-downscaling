import os
import yaml
import logging
import torch
import gpytorch
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from progress.bar import Bar
from models import VariationalCMEProcess, CMEProcessLikelihood, RFFKernel, MODELS, TRAINERS, PREDICTERS
from core.visualization import plot_downscaling_prediction
from core.metrics import compute_metrics


@MODELS.register('variational_cme_process')
def build_downscaling_variational_cme_process(covariates_grid, lbda, n_inducing_points, use_individuals_noise, seed, **kwargs):
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
    base_indiv_spatial_kernel = RFFKernel(nu=1.5, num_samples=1000, ard_num_dims=3, active_dims=[0, 1, 2])
    base_indiv_spatial_kernel.initialize(raw_lengthscale=inv_softplus(x=1, n=3))

    base_indiv_feat_kernel = gpytorch.kernels.RFFKernel(num_samples=1000, ard_num_dims=3, active_dims=[3, 4, 5])
    base_indiv_feat_kernel.initialize(raw_lengthscale=inv_softplus(x=1, n=3))

    individuals_spatial_kernel = gpytorch.kernels.ScaleKernel(base_indiv_spatial_kernel)
    individuals_feat_kernel = gpytorch.kernels.ScaleKernel(base_indiv_feat_kernel)
    individuals_kernel = individuals_spatial_kernel + individuals_feat_kernel

    # Define bags kernels
    base_bag_spatial_kernel = gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=3, active_dims=[0, 1, 2])
    base_bag_spatial_kernel.initialize(raw_lengthscale=inv_softplus(x=1, n=3))

    base_bag_feat_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=3, active_dims=[3, 4, 5])
    base_bag_feat_kernel.initialize(raw_lengthscale=inv_softplus(x=1, n=3))

    bag_spatial_kernel = gpytorch.kernels.ScaleKernel(base_bag_spatial_kernel)
    bag_feat_kernel = gpytorch.kernels.ScaleKernel(base_bag_feat_kernel)
    bag_kernel = bag_spatial_kernel + bag_feat_kernel

    # Initialize inducing points with kmeans
    if seed:
        torch.random.manual_seed(seed)
    flattened_grid = covariates_grid.view(-1, covariates_grid.size(-1))
    rdm_idx = torch.randperm(flattened_grid.size(0))[:n_inducing_points]
    inducing_points = flattened_grid[rdm_idx].float()
    # logging.info(f"K-Means selection of {n_inducing_points} inducing points with seed {seed}")
    # kmeans = KMeans(n_clusters=n_inducing_points, init='k-means++', random_state=seed)
    # kmeans.fit(covariates_grid.view(-1, covariates_grid.size(-1)))
    # inducing_points = torch.from_numpy(kmeans.cluster_centers_).float()

    # Define model
    model = VariationalCMEProcess(individuals_mean=individuals_mean,
                                  individuals_kernel=individuals_kernel,
                                  bag_kernel=bag_kernel,
                                  inducing_points=inducing_points,
                                  lbda=lbda,
                                  use_individuals_noise=use_individuals_noise)
    return model


@TRAINERS.register('variational_cme_process')
def train_downscaling_variational_cme_process(model, covariates_blocks, bags_blocks, extended_bags, targets_blocks,
                                              lr, n_epochs, batch_size, beta, seed, dump_dir, covariates_grid,
                                              use_individuals_noise, groundtruth_field, target_field, plot, plot_every, **kwargs):
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
    extended_bags = extended_bags.to(device)
    targets_blocks = targets_blocks.to(device)

    ################################################################ Drop certain bags
    n_drop = len(targets_blocks) // 2
    torch.random.manual_seed(5)
    rdm_indices = torch.randperm(len(targets_blocks)).to(device)

    paired_covariates_blocks = covariates_blocks[rdm_indices[n_drop:]]
    paired_bags_blocks = bags_blocks[rdm_indices[n_drop:]]
    paired_extended_bags = extended_bags[rdm_indices[n_drop:]]
    paired_targets_blocks = targets_blocks[rdm_indices[n_drop:]]

    n_dim_individuals = covariates_blocks.size(-1)
    n_dim_bags = bags_blocks.size(-1)
    leftovers_covariates = covariates_blocks[rdm_indices[:n_drop]].reshape(-1, n_dim_individuals)
    leftovers_extended_bags = extended_bags[rdm_indices[:n_drop]].reshape(-1, n_dim_bags)

    def leftovers_iterator(batch_size):
        buffer = torch.ones(len(leftovers_covariates)).to(device)
        while True:
            idx = buffer.multinomial(batch_size)
            x = leftovers_covariates[idx]
            extended_y = leftovers_extended_bags[idx]
            yield x, extended_y

    def batch_iterator(batch_size):
        rdm_indices = torch.randperm(len(paired_targets_blocks)).to(device)
        n_dim_individuals = covariates_blocks.size(-1)
        n_dim_bags = bags_blocks.size(-1)
        unpaired_samples_sampler = leftovers_iterator(batch_size * covariates_blocks.size(1))
        for idx in rdm_indices.split(batch_size):
            x = paired_covariates_blocks[idx].reshape(-1, n_dim_individuals)
            y = paired_bags_blocks[idx]
            extended_y = paired_extended_bags[idx].reshape(-1, n_dim_bags)
            z = paired_targets_blocks[idx]
            leftovers_x, leftovers_extended_y = next(unpaired_samples_sampler)
            x = torch.cat([x, leftovers_x])
            extended_y = torch.cat([extended_y, leftovers_extended_y])
            yield x, y, extended_y, z
    ################################################################################################

    # # Define stochastic batch iterator
    # def batch_iterator(batch_size):
    #     rdm_indices = torch.randperm(len(targets_blocks)).to(device)
    #     n_dim_individuals = covariates_blocks.size(-1)
    #     n_dim_bags = bags_blocks.size(-1)
    #     for idx in rdm_indices.split(batch_size):
    #         x = covariates_blocks[idx].reshape(-1, n_dim_individuals)
    #         y = bags_blocks[idx]
    #         extended_y = extended_bags[idx].reshape(-1, n_dim_bags)
    #         z = targets_blocks[idx]
    #         yield x, y, extended_y, z

    # Define variational CME process likelihood
    likelihood = CMEProcessLikelihood(use_individuals_noise=use_individuals_noise)

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

    # Logs record
    logs = dict()

    for epoch in range(n_epochs):

        batch_bar = Bar("Batch", max=len(targets_blocks) // batch_size)

        for x, y, extended_y, z in batch_iterator(batch_size):
            # Zero-out remaining gradients
            optimizer.zero_grad()

            # Compute q(f)
            q = model(x)

            # Compute tensors needed for ELBO computation
            elbo_kwargs = model.get_elbo_computation_parameters(bags_values=y,
                                                                extended_bags_values=extended_y)

            # Compute negative ELBO loss
            loss = -elbo(variational_dist_f=q,
                         target=z,
                         **elbo_kwargs)

            # Take gradient step
            loss.backward()
            optimizer.step()

            batch_bar.suffix = f"ELBO {-loss.item()}"
            batch_bar.next()

        # Compute posterior distribution at current epoch and store logs
        individuals_posterior = predict_downscaling_variational_cme_process(model=model,
                                                                            covariates_grid=covariates_grid,
                                                                            mean_shift=mean_shift,
                                                                            std_scale=std_scale)
        epoch_logs = get_epoch_logs(model, likelihood, individuals_posterior, groundtruth_field)
        logs[epoch + 1] = epoch_logs
        with open(os.path.join(dump_dir, 'running_logs.yaml'), 'w') as f:
            yaml.dump({'epoch': logs}, f)

        # Dump plot of posterior prediction at current epoch
        if plot and epoch % plot_every == 0:
            _ = plot_downscaling_prediction(individuals_posterior, groundtruth_field, target_field)
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
    k_spatial_kernel, k_feat_kernel = model.individuals_kernel.kernels
    k_spatial_lengthscales = k_spatial_kernel.base_kernel.lengthscale[0].detach().tolist()
    k_feat_lengthscales = k_feat_kernel.base_kernel.lengthscale[0].detach().tolist()

    l_spatial_kernel, l_feat_kernel = model.bag_kernel.kernels
    l_spatial_lengthscales = l_spatial_kernel.base_kernel.lengthscale[0].detach().tolist()
    l_feat_lengthscales = l_feat_kernel.base_kernel.lengthscale[0].detach().tolist()

    epoch_logs.update({'aggregate_noise': likelihood.noise.detach().item(),
                       'k_spatial_outputscale': k_spatial_kernel.outputscale.detach().item(),
                       'k_lengthscale_lat': k_spatial_lengthscales[0],
                       'k_lengthscale_lon': k_spatial_lengthscales[1],
                       'k_lengthscale_alt': k_spatial_lengthscales[2],
                       'k_feat_outputscale': k_feat_kernel.outputscale.detach().item(),
                       'k_lengthscale_albisccp': k_feat_lengthscales[0],
                       'k_lengthscale_clt': k_feat_lengthscales[1],
                       'k_lengthscale_pctisccp': k_feat_lengthscales[2],
                       'l_spatial_outputscale': l_spatial_kernel.outputscale.detach().item(),
                       'l_lengthscale_lat': l_spatial_lengthscales[0],
                       'l_lengthscale_lon': l_spatial_lengthscales[1],
                       'l_lengthscale_alt': l_spatial_lengthscales[2],
                       'l_feat_outputscale': l_feat_kernel.outputscale.detach().item(),
                       'l_lengthscale_albisccp': l_feat_lengthscales[0],
                       'l_lengthscale_clt': l_feat_lengthscales[1],
                       'l_lengthscale_pctisccp': l_feat_lengthscales[2]})
    if model.noise_kernel:
        epoch_logs.update({'indiv_noise': model.noise_kernel.outputscale.detach().item()})
    return epoch_logs


@PREDICTERS.register('variational_cme_process')
def predict_downscaling_variational_cme_process(model, covariates_grid, mean_shift, std_scale, **kwargs):
    # Set model in evaluation mode
    model.eval()

    # Compute standardized posterior distribution on individuals
    with torch.no_grad():
        logging.info("\n Infering deconditioning posterior on HR pixels...")
        individuals_posterior = model(covariates_grid.view(-1, covariates_grid.size(-1)))

    # Rescale by mean and std from observed aggregate target field
    mean_posterior = mean_shift + std_scale * individuals_posterior.mean.cpu()
    lazy_covariance_posterior = (std_scale**2) * individuals_posterior.lazy_covariance_matrix.cpu()
    individuals_posterior = gpytorch.distributions.MultivariateNormal(mean=mean_posterior,
                                                                      covariance_matrix=lazy_covariance_posterior)

    # Set model back to training mode
    model.train()
    return individuals_posterior
