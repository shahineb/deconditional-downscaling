import torch
import matplotlib.pyplot as plt


def plot_dataset(individuals, groundtruth_individuals, targets, aggregate_targets, bags_sizes, scatter_kwargs={}):
    """Plots altogether views of the dataset to visualize how was it built

    Args:
        individuals (torch.Tensor)
        groundtruth_individuals (torch.Tensor)
        targets (torch.Tensor)
        aggregate_targets (torch.Tensor)
        bags_sizes (list[int])
        scatter_kwargs (dict)

    Returns:
        type: matplotlib.figure.Figure

    """
    # Instantiate figure and default plotting arguments
    fig = plt.figure(figsize=(20, 20))
    view_angle = (10, -100)
    scatter_kwargs.update({'xs': individuals[:, 0],
                           'ys': individuals[:, 1],
                           'zs': individuals[:, 2],
                           'alpha': 1,
                           's': 5})

    # First - individuals we have access to for CMO estimation
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    ax.view_init(*view_angle)
    ax.scatter(**scatter_kwargs,
               c='k')
    ax.set_title("Swiss roll individuals positions used for training", fontsize=24)

    # Second - used bagging of the latter individuals
    ax = fig.add_subplot(2, 2, 2, projection='3d')
    ax.view_init(10, -100)
    for x in individuals.split(bags_sizes):
        ax.scatter(x[:, 0],
                   x[:, 1],
                   x[:, 2],
                   s=5)
    ax.set_title("Vertical bagging of individuals", fontsize=24)

    # Third - groundtruth individuals from which aggregate targets are built
    ax = fig.add_subplot(2, 2, 3, projection='3d')
    ax.view_init(*view_angle)
    ax.scatter(groundtruth_individuals[:, 0],
               groundtruth_individuals[:, 1],
               groundtruth_individuals[:, 2],
               c=targets,
               cmap='Spectral',
               s=5)
    ax.set_title("Swiss roll samples from groundtruth distribution", fontsize=24)

    # Fourth - resulting aggregate targets mapped to bags from 2nd plot
    expanded_aggregate_targets = torch.cat([x.unsqueeze(0).repeat(bag_size) for x, bag_size in zip(aggregate_targets, bags_sizes)])
    ax = fig.add_subplot(2, 2, 4, projection='3d')
    ax.view_init(10, -100)
    ax.scatter(**scatter_kwargs,
               c=expanded_aggregate_targets,
               cmap='Spectral')
    ax.set_title("Bags colored by Aggregate Target", fontsize=24)
    return fig


def plot_grountruth_prediction(individuals_posterior, groundtruth_individuals, targets, scatter_kwargs={}):
    """Plots mean prediction, grountruth, error and confidence region size for each sample

    Args:
        individuals (torch.Tensor)
        groundtruth_individuals (torch.Tensor)
        targets (torch.Tensor)
        scatter_kwargs (dict)

    Returns:
        type: matplotlib.figure.Figure

    """
    # Compute mutual min/max of prediction and grountruth for rescaled color plotting
    max_value = torch.max(torch.stack([targets.max(),
                                       individuals_posterior.mean.max()])).item()
    min_value = torch.min(torch.stack([targets.min(),
                                       individuals_posterior.mean.min()])).item()

    # Instantiate figure and default plotting arguments
    fig = plt.figure(figsize=(20, 20))
    view_angle = (10, -100)
    scatter_kwargs.update({'xs': groundtruth_individuals[:, 0],
                           'ys': groundtruth_individuals[:, 1],
                           'zs': groundtruth_individuals[:, 2],
                           'alpha': 1,
                           's': 5})

    # First - mean posterior prediction
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    ax.view_init(*view_angle)
    ax.scatter(c=individuals_posterior.mean.numpy(),
               cmap='Spectral',
               vmin=min_value,
               vmax=max_value,
               **scatter_kwargs)
    ax.set_title("Mean Posterior Prediction", fontsize=24)

    # Second - grountruth
    ax = fig.add_subplot(2, 2, 2, projection='3d')
    ax.view_init(*view_angle)
    roll = ax.scatter(c=targets.numpy(),
                      cmap='Spectral',
                      vmin=min_value,
                      vmax=max_value,
                      **scatter_kwargs)
    cb = fig.colorbar(roll, shrink=0.5, ax=ax)
    cb.ax.tick_params(labelsize=14)
    ax.set_title("Groundtruth", fontsize=24)

    # Third - sample-wise squared error
    ax = fig.add_subplot(2, 2, 3, projection='3d')
    square_error = torch.pow(targets - individuals_posterior.mean, 2)
    ax.view_init(*view_angle)
    roll = ax.scatter(c=square_error.numpy(),
                      cmap='inferno',
                      **scatter_kwargs)
    cb = fig.colorbar(roll, shrink=0.5, ax=ax)
    cb.ax.tick_params(labelsize=14)
    ax.set_title("Squared Error", fontsize=24)

    # Fourth - sample-wise confidence region size
    ax = fig.add_subplot(2, 2, 4, projection='3d')
    conf = individuals_posterior.confidence_region()
    conf_size = conf[1] - conf[0]
    ax.view_init(*view_angle)
    roll = ax.scatter(c=conf_size.detach().numpy(),
                      cmap='inferno',
                      **scatter_kwargs)
    cb = fig.colorbar(roll, shrink=0.5, ax=ax)
    cb.ax.tick_params(labelsize=14)
    ax.set_title("Confidence Region size (±2σ)", fontsize=24)
    return fig
