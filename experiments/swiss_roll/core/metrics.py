import numpy as np
import torch
import gpytorch


def compute_metrics(individuals_posterior, groundtruth_targets):
    """Computes MSE, MAE, Mean Bias, Spearman Correlation and SSIM between downscaled
    field and groundtruth and dump into YAML metrics file
    """
    mse = torch.pow(individuals_posterior.mean - groundtruth_targets, 2).mean()
    mae = torch.abs(individuals_posterior.mean - groundtruth_targets).mean()
    mb = torch.mean(individuals_posterior.mean - groundtruth_targets)
    individuals_metrics = {'mse': mse.item(),
                           'mae': mae.item(),
                           'mb': mb.item()}
    return individuals_metrics


def compute_chunked_nll(model, predict, groundtruth_individuals, groundtruth_targets, chunk_size):
    """Compute NLL on chunks of dataset for scalability issue and averages chunks NLL
    """
    try:
        nlls = []
        with gpytorch.settings.fast_computations(log_prob=False, covar_root_decomposition=False):
            for X, t in zip(groundtruth_individuals.split(chunk_size), groundtruth_targets.split(chunk_size)):
                sub_posterior = predict(model=model, individuals=X)
                nll = -sub_posterior.log_prob(t).div(chunk_size)
                nlls.append(nll.item())
        output = float(np.mean(nlls))
    except gpytorch.utils.errors.NotPSDError:
        output = np.nan
    return output
