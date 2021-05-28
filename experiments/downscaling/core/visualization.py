import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap


def plot_downscaling_prediction(individuals_posterior, groundtruth_field, target_field, drop_idx=None):
    # Prepare fields to plot
    groundtruth = torch.from_numpy(groundtruth_field.values)
    mean_pred = individuals_posterior.mean.reshape(*groundtruth.shape).cpu()
    difference = groundtruth - mean_pred
    squared_error = difference.pow(2)
    conf = individuals_posterior.confidence_region()
    conf_size = conf[1] - conf[0]
    conf_size = conf_size.reshape(*mean_pred.shape).cpu()

    # Get values range for prediction and difference
    max_value = max(torch.max(torch.stack([groundtruth.max(), mean_pred.max()])).item(), target_field.values.max())
    min_value = min(torch.min(torch.stack([groundtruth.min(), mean_pred.min()])).item(), target_field.values.min())
    diff_abs_max = difference.abs().max().item()

    # If specified, artificially mask out dropped bags
    if drop_idx is not None:
        buffer = target_field.values.flatten()
        buffer[drop_idx.tolist()] = min_value - (max_value - min_value) / 254
        target_field.values = buffer.reshape(*target_field.shape)
        min_value = target_field.values.min()

    # Create custom colormap that will set unobserved pixels in grey
    field_cmap = cm.get_cmap('magma', 256)
    field_cmap_colors = field_cmap(np.linspace(0, 1, 255))
    grey = np.array([[126, 126, 126, 256]]) / 256
    extended_field_cmap_colors = np.concatenate([grey, field_cmap_colors])
    extended_field_cmap = ListedColormap(extended_field_cmap_colors)

    # Instantiate plot and layout params
    fig, ax = plt.subplots(2, 3, figsize=(48, 22))
    uncertainty_cmap = 'turbo'
    squared_error_cmap = 'CMRmap'
    bias_cmap = 'bwr'
    fontsize = 32

    # Plot observed LR field
    im = ax[0, 0].imshow(target_field.values[::-1], cmap=extended_field_cmap, vmin=min_value, vmax=max_value)
    ax[0, 0].axis('off')
    ax[0, 0].set_title("Observed LR field", fontsize=fontsize)
    cb = fig.colorbar(im, ax=ax[0, 0], shrink=0.5)
    cb.ax.tick_params(labelsize=fontsize)
    cb.ax.set_ylabel("T (°K)", rotation=90, fontsize=fontsize)

    # Plot mean posterior prediction
    im = ax[0, 1].imshow(mean_pred.numpy()[::-1], cmap=field_cmap, vmin=min_value, vmax=max_value)
    ax[0, 1].axis('off')
    ax[0, 1].set_title("Mean Posterior Prediction", fontsize=fontsize)
    cb = fig.colorbar(im, ax=ax[0, 1], shrink=0.5)
    cb.ax.tick_params(labelsize=fontsize)
    cb.ax.set_ylabel("T (°K)", rotation=90, fontsize=fontsize)

    # Plot 95% confidence region size
    im = ax[0, 2].imshow(conf_size.numpy()[::-1], cmap=uncertainty_cmap)
    ax[0, 2].axis('off')
    ax[0, 2].set_title("Confidence Region Size (±2σ)", fontsize=fontsize)
    cb = fig.colorbar(im, ax=ax[0, 2], shrink=0.5)
    cb.ax.tick_params(labelsize=fontsize)
    cb.ax.set_ylabel("T (°K)", rotation=90, fontsize=fontsize)

    # Plot unobserved groundtruth HR field
    im = ax[1, 0].imshow(groundtruth.numpy()[::-1], cmap=field_cmap, vmin=min_value, vmax=max_value)
    ax[1, 0].axis('off')
    ax[1, 0].set_title("Unobserved groundtruth HR field", fontsize=fontsize)
    cb = fig.colorbar(im, ax=ax[1, 0], shrink=0.5)
    cb.ax.tick_params(labelsize=fontsize)
    cb.ax.set_ylabel("T (°K)", rotation=90, fontsize=fontsize)

    # Plot pixelwise squared error
    im = ax[1, 1].imshow(squared_error.numpy()[::-1], cmap=squared_error_cmap)
    ax[1, 1].axis('off')
    ax[1, 1].set_title("Squared Error", fontsize=fontsize)
    cb = fig.colorbar(im, ax=ax[1, 1], shrink=0.5)
    cb.ax.tick_params(labelsize=fontsize)

    # Plot pixelwise bias
    im = ax[1, 2].imshow(difference.numpy()[::-1], cmap=bias_cmap, vmin=-diff_abs_max, vmax=diff_abs_max)
    ax[1, 2].axis('off')
    ax[1, 2].set_title("(Grountruth) – (Mean Posterior)", fontsize=fontsize)
    cb = fig.colorbar(im, ax=ax[1, 2], shrink=0.5)
    cb.ax.tick_params(labelsize=fontsize)
    cb.ax.set_ylabel("T (°K)", rotation=90, fontsize=fontsize)

    plt.tight_layout()
    return fig
