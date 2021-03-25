import torch
import matplotlib.pyplot as plt
from core.metrics import compute_metrics


def plot_downscaling_prediction(individuals_posterior, groundtruth_field, target_field):
    individuals_metrics = compute_metrics(individuals_posterior, groundtruth_field)
    metrics_str = ' | '.join([key + " : " + str(round(value, 3)) for (key, value) in individuals_metrics.items()])

    groundtruth = torch.from_numpy(groundtruth_field.values)
    posterior_mean = individuals_posterior.mean.view(*groundtruth.shape)
    difference = posterior_mean - groundtruth

    max_value = torch.max(torch.stack([groundtruth.max(),
                                       posterior_mean.max()])).item()
    min_value = torch.min(torch.stack([groundtruth.min(),
                                       posterior_mean.min()])).item()
    diff_abs_max = difference.abs().max().item()

    fig, ax = plt.subplots(2, 2, figsize=(30, 18))
    im = ax[0, 0].imshow(groundtruth.numpy()[::-1], cmap='magma', vmin=min_value, vmax=max_value)
    fig.colorbar(im, ax=ax[0, 0], shrink=0.5)
    ax[0, 0].set_title("Unobserved Groundtruth HR Cloud Top Temperature")

    im = ax[0, 1].imshow(posterior_mean.numpy()[::-1], cmap='magma', vmin=min_value, vmax=max_value)
    fig.colorbar(im, ax=ax[0, 1], shrink=0.5)
    ax[0, 1].set_title("Mean Posterior Prediction")

    im = ax[1, 0].imshow(target_field.values[::-1], cmap='magma')
    fig.colorbar(im, ax=ax[1, 0], shrink=0.5)
    ax[1, 0].set_title("Observed LR Cloud Top Temperature")

    im = ax[1, 1].imshow(difference.numpy()[::-1], cmap='bwr', vmin=-diff_abs_max, vmax=diff_abs_max)
    fig.colorbar(im, ax=ax[1, 1], shrink=0.5)
    ax[1, 1].set_title("Mean Posterior - Groundtruth | " + metrics_str)

    plt.tight_layout()
    return fig
