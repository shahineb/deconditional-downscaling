import os
import yaml
import logging
import torch
import gpytorch
import matplotlib.pyplot as plt
from progress.bar import Bar
from models import VariationalGP, RFFKernel, ExactGP, MODELS, TRAINERS, PREDICTERS
from core.visualization import plot_downscaling_prediction
from core.metrics import compute_metrics
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']


@MODELS.register('krigging')
def build_downscaling_variational_krigging(covariates_grid, n_inducing_points, seed, **kwargs):
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
    base_spatial_kernel = RFFKernel(nu=1.5, num_samples=1000, ard_num_dims=2, active_dims=[0, 1])
    base_spatial_kernel.initialize(raw_lengthscale=inv_softplus(x=1, n=2))

    base_feat_kernel = gpytorch.kernels.RFFKernel(num_samples=1000, ard_num_dims=3, active_dims=[2, 3, 4])
    base_feat_kernel.initialize(raw_lengthscale=inv_softplus(x=1, n=3))

    spatial_kernel = gpytorch.kernels.ScaleKernel(base_spatial_kernel)
    feat_kernel = gpytorch.kernels.ScaleKernel(base_feat_kernel)
    covar_module = spatial_kernel + feat_kernel

    # Initialize inducing points regularly across grid
    flattened_grid = covariates_grid.view(-1, covariates_grid.size(-1))
    n_samples = flattened_grid.size(0)
    step = n_samples // n_inducing_points
    offset = (n_samples % n_inducing_points) // 2
    inducing_points = flattened_grid[offset:n_samples - offset:step].float()

    # Define model
    model = VariationalGP(inducing_points=inducing_points,
                          mean_module=mean_module,
                          covar_module=covar_module)
    return model


def fit_gp_regression(bags_blocks, targets_blocks, device):
    # Inverse softplus utility for gpytorch lengthscale intialization
    inv_softplus = lambda x, n: torch.log(torch.exp(x * torch.ones(n)) - 1)

    # Define mean and covariance modules
    bag_mean = gpytorch.means.ZeroMean()

    # Define bags kernels
    base_bag_spatial_kernel = gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=2, active_dims=[0, 1])
    base_bag_spatial_kernel.initialize(raw_lengthscale=inv_softplus(x=1, n=2))

    base_bag_feat_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=1, active_dims=[2])
    base_bag_feat_kernel.initialize(raw_lengthscale=inv_softplus(x=1, n=1))

    bag_spatial_kernel = gpytorch.kernels.ScaleKernel(base_bag_spatial_kernel)
    bag_feat_kernel = gpytorch.kernels.ScaleKernel(base_bag_feat_kernel)
    bag_kernel = bag_spatial_kernel + bag_feat_kernel

    # Define model
    model = ExactGP(mean_module=bag_mean,
                    covar_module=bag_kernel,
                    train_x=bags_blocks,
                    train_y=targets_blocks,
                    likelihood=gpytorch.likelihoods.GaussianLikelihood())

    # Set model in training mode
    model = model.train().to(device)

    # Define optimizer and exact loglikelihood module
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.2)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    # Initialize progress bar
    bar = Bar("Epoch", max=400)

    for epoch in range(400):
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


@TRAINERS.register('krigging')
def train_downscaling_variational_krigging(model, covariates_blocks, bags_blocks, targets_blocks,
                                           lr, n_epochs, batch_size, beta, seed, device_idx, dump_dir, covariates_grid, fill_missing,
                                           groundtruth_field, target_field, missing_bags_fraction, plot, plot_every, log_every, **kwargs):
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
    device = torch.device(f"cuda:{device_idx}") if torch.cuda.is_available() else torch.device("cpu")
    covariates_grid = covariates_grid.to(device)
    covariates_blocks = covariates_blocks.to(device)
    bags_blocks = bags_blocks.to(device)
    targets_blocks = targets_blocks.to(device)

    # Split dataset in unmatched sets
    if seed:
        torch.random.manual_seed(seed)
    n_drop = int(missing_bags_fraction * len(targets_blocks))
    shuffled_indices = torch.randperm(len(targets_blocks)).to(device)
    indices_1, indices_2 = shuffled_indices[n_drop:], shuffled_indices[:n_drop]

    regressor = fit_gp_regression(bags_blocks[indices_2].cpu(), targets_blocks[indices_2].cpu(), device)
    with torch.no_grad():
        aggregate_covariates_blocks = covariates_blocks[indices_1].mean(dim=1)
        targets_blocks = regressor(bags_blocks[indices_1]).mean

    # Define stochastic batch iterator
    def batch_iterator(batch_size):
        rdm_indices = torch.randperm(len(targets_blocks)).to(device)
        for idx in rdm_indices.split(batch_size):
            x = aggregate_covariates_blocks[idx]
            z = targets_blocks[idx]
            yield x, z

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

        for x, z in batch_iterator(batch_size):
            # Zero-out remaining gradients
            optimizer.zero_grad()

            # Compute q(f)
            q = model(x)

            # Compute negative ELBO loss
            loss = -elbo(variational_dist_f=q, target=z)

            # Take gradient step
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_bar.suffix = f"Running ELBO {-loss.item()}"
            batch_bar.next()

        if epoch % log_every == 0:
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
                _ = plot_downscaling_prediction(individuals_posterior, groundtruth_field, target_field, indices_1)
                plt.savefig(os.path.join(dump_dir, f'png/epoch_{epoch}.png'))
                plt.close()
            epoch_bar.next()
            epoch_bar.finish()

            # Empty cache if using GPU
            if torch.cuda.is_available():
                with torch.cuda.device(f"cuda:{device_idx}"):
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
                       'k_feat_outputscale': k_feat_kernel.outputscale.detach().item(),
                       'k_lengthscale_alt': k_feat_lengthscales[0],
                       'k_lengthscale_albisccp': k_feat_lengthscales[1],
                       'k_lengthscale_clt': k_feat_lengthscales[2]})
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


def levenshtein(string_1, string_2):
    rows = len(string_1) + 1
    cols = len(string_2) + 1
    dist = [[0 for x in range(cols)] for x in range(rows)]

    for i in range(1, rows):
        dist[i][0] = i
    for i in range(1, cols):
        dist[0][i] = i

    for col in range(1, cols):
        for row in range(1, rows):
            if string_1[row - 1] == string_2[col - 1]:
                cost = 0
            else:
                cost = 1
            dist[row][col] = min(dist[row - 1][col] + 1,
                                 dist[row][col - 1] + 1,
                                 dist[row - 1][col - 1] + cost)
    return dist[row][col]
