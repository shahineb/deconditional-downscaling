import torch


def compute_metrics(individuals_posterior, groundtruth):
    """Computes MSE, MAE, Mean Bias, Spearman Correlation and SSIM between downscaled
    field and groundtruth and dump into YAML metrics file
    """
    mse = torch.pow(individuals_posterior.mean - groundtruth, 2).mean()
    mae = torch.abs(individuals_posterior.mean - groundtruth).mean()
    mb = torch.mean(individuals_posterior.mean - groundtruth)
    individuals_metrics = {'mse': mse.item(),
                           'mae': mae.item(),
                           'mb': mb.item()}
    return individuals_metrics
