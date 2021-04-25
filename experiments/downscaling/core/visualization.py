import torch
import matplotlib.pyplot as plt


def plot_downscaling_prediction(individuals_posterior, groundtruth_field, target_field):
    # Prepare fields to plot
    groundtruth = torch.from_numpy(groundtruth_field.values)
    mean_pred = individuals_posterior.mean.reshape(*groundtruth.shape).cpu()
    difference = groundtruth - mean_pred
    squared_error = difference.pow(2)
    conf = individuals_posterior.confidence_region()
    conf_size = conf[1] - conf[0]
    conf_size = conf_size.reshape(*mean_pred.shape).cpu()

    # Get values range for prediction and difference
    max_value = torch.max(torch.stack([groundtruth.max(),
                                       mean_pred.max()])).item()
    min_value = torch.min(torch.stack([groundtruth.min(),
                                       mean_pred.min()])).item()
    diff_abs_max = difference.abs().max().item()

    # Instantiate plot and layout params
    fig, ax = plt.subplots(2, 3, figsize=(44, 18))
    field_cmap = 'magma'
    uncertainty_cmap = 'turbo'
    squared_error_cmap = 'CMRmap'
    bias_cmap = 'bwr'
    fontsize = 24

    # Plot observed LR field
    im = ax[0, 0].imshow(target_field.values[::-1], cmap=field_cmap, vmin=min_value, vmax=max_value)
    ax[0, 0].axis('off')
    ax[0, 0].set_title("Observed LR field", fontsize=fontsize)
    fig.colorbar(im, ax=ax[0, 0], shrink=0.5)

    # Plot mean posterior prediction
    im = ax[0, 1].imshow(mean_pred.numpy()[::-1], cmap=field_cmap, vmin=min_value, vmax=max_value)
    ax[0, 1].axis('off')
    ax[0, 1].set_title("Mean Posterior Prediction", fontsize=fontsize)
    fig.colorbar(im, ax=ax[0, 1], shrink=0.5)

    # Plot 95% confidence region size
    im = ax[0, 2].imshow(conf_size.numpy()[::-1], cmap=uncertainty_cmap)
    ax[0, 2].axis('off')
    ax[0, 2].set_title("Confidence Region Size (±2σ)", fontsize=fontsize)
    fig.colorbar(im, ax=ax[0, 2], shrink=0.5)

    # Plot unobserved groundtruth HR field
    im = ax[1, 0].imshow(groundtruth.numpy()[::-1], cmap=field_cmap, vmin=min_value, vmax=max_value)
    ax[1, 0].axis('off')
    ax[1, 0].set_title("Unobserved groundtruth HR field", fontsize=fontsize)
    fig.colorbar(im, ax=ax[1, 0], shrink=0.5)

    # Plot pixelwise squared error
    im = ax[1, 1].imshow(squared_error.numpy()[::-1], cmap=squared_error_cmap)
    ax[1, 1].axis('off')
    ax[1, 1].set_title("Squared Error", fontsize=fontsize)
    fig.colorbar(im, ax=ax[1, 1], shrink=0.5)

    # Plot pixelwise bias
    im = ax[1, 2].imshow(difference.numpy()[::-1], cmap=bias_cmap, vmin=-diff_abs_max, vmax=diff_abs_max)
    ax[1, 2].axis('off')
    ax[1, 2].set_title("(Grountruth) – (Mean Posterior)", fontsize=fontsize)
    fig.colorbar(im, ax=ax[1, 2], shrink=0.5)

    plt.tight_layout()
    return fig
