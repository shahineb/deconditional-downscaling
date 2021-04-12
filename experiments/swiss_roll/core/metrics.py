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


def compute_subsampled_nll(X_gt, t_gt, n_individuals, model, predict, seed=None):
    # Select subset of individuals for NLL computation - scalability
    if seed:
        torch.random.manual_seed(seed)
    rdm_idx = torch.randperm(X_gt.size(0))
    sub_individuals = X_gt[rdm_idx][:n_individuals]
    sub_individuals_target = t_gt[rdm_idx][:n_individuals]

    # Compute model NLL on subset
    with gpytorch.settings.fast_computations(log_prob=False, covar_root_decomposition=False):
        sub_individuals_posterior = predict(model=model, individuals=sub_individuals)
        nll = -sub_individuals_posterior.log_prob(sub_individuals_target).div(n_individuals)
    return nll
