import numpy as np
import torch
import gpytorch


def compute_metrics(individuals_posterior, groundtruth):
    """Computes MSE, MAE, Mean Bias, Spearman Correlation and SSIM between downscaled
    field and groundtruth and dump into YAML metrics file
    """
    with torch.no_grad():
        mse = torch.pow(individuals_posterior.mean - groundtruth, 2).mean()
        mae = torch.abs(individuals_posterior.mean - groundtruth).mean()
        mb = torch.mean(individuals_posterior.mean - groundtruth)
        individuals_metrics = {'mse': mse.item(),
                               'mae': mae.item(),
                               'mb': mb.item()}
        return individuals_metrics


def compute_chunked_nll(X_gt, t_gt, chunk_size, model, predict):
    # Compute model NLL on chunks
    nlls = []
    with gpytorch.settings.fast_computations(log_prob=False, covar_root_decomposition=False):
        for sub_X, sub_t in zip(X_gt.split(chunk_size), t_gt.split(chunk_size)):
            sub_posterior = predict(model=model, individuals=sub_X)
            nll = -sub_posterior.log_prob(sub_t).div(chunk_size)
            nlls.append(nll.item())
    return np.mean(nlls)
