import numpy as np
import xarray as xr
import torch


def load_fields(files_paths, date=None):
    """Loads fields at specified path and date.
    Returns them in dictionnary mapping field name to field

    Args:
        files_paths (list[str]): paths to .nc files to load
        date (str): date formatted as 'yyyy-mm-dd'

    Returns:
        type: dict[xarray.core.dataarray.DataArray]

    """
    fields = [load_field(file_path, date) for file_path in files_paths]
    fields = {x.name: x for x in fields}
    return fields


def preprocess_fields(fields, covariate_fields_names, target_field_name, block_size):
    """
        (1): Standardizes all fields
        (2): Downsamples target field to target resolution
        (3): Trims covariate fields to a size multiple of downsampling block

    Args:
        fields (dict[xarray.core.dataarray.DataArray])
        covariate_fields_names (list[str]): names of fields used as covariates
        target_field_name (str): list of field used as aggregate target
        block_size (tuple[int]): (height, width) dimensions of blocks to average

    Returns:
        type: dict[xarray.core.dataarray.DataArray], xarray.core.dataarray.DataArray

    """
    # Coarsen and standardize aggregate target field
    raw_aggregate_target_field = coarsen(field=fields[target_field_name], block_size=block_size)
    aggregate_target_field = standardize(field=raw_aggregate_target_field)

    # Standardize and trim covariates fields to match dimensions of target field
    covariates_fields = dict()
    for key in covariate_fields_names:
        standardized_field = standardize(field=fields[key])
        trimmed_field = trim(field=standardized_field, block_size=block_size)
        covariates_fields.update({key: trimmed_field})

    return covariates_fields, aggregate_target_field, raw_aggregate_target_field


def make_tensor_dataset(covariates_fields, aggregate_target_field, block_size):
    """Converts covariates and target fields into bagged pytorch tensors

    Args:
        covariates_fields (dict[xarray.core.dataarray.DataArray])
        aggregate_target_field (xarray.core.dataarray.DataArray)
        block_size (tuple[int]): (height, width) dimensions of blocks to average

    Returns:
        type: torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor

    """
    covariates_grid = make_covariates_grid_tensor(list(covariates_fields.values()))
    bags_grid = make_lat_lon_grid_tensor(aggregate_target_field)
    target_grid = torch.from_numpy(aggregate_target_field.values)
    covariates_blocks, bags_blocks, extended_bags, targets_blocks = make_bagged_dataset(covariates_grid=covariates_grid,
                                                                                        bags_grid=bags_grid,
                                                                                        target_grid=target_grid,
                                                                                        block_size=block_size)
    return covariates_grid.float(), covariates_blocks.float(), bags_blocks.float(), extended_bags.float(), targets_blocks.float()


def load_field(file_path, date=None):
    """Loads xarray Dataset, selects slice corresponding to specified date
        and extracts DataArray of Dataset main variable

    Args:
        file_path (str): path to .nc file to load
        date (str): date formatted as 'yyyy-mm-dd'

    Returns:
        type: xarray.core.dataarray.DataArray

    """
    dataset = xr.load_dataset(file_path)
    try:
        dataset = dataset.sel(time=date)
    except ValueError:
        pass
    field = list(dataset.values()).pop()
    return field


def coarsen(field, block_size):
    """Downsamples dataarray along latitude and longitude according to xarray
        coarsening method. If not multiple of coarsening block size, borders are
        trimmed.

    Args:
        field (xarray.core.dataarray.DataArray)
        block_size (tuple[int]): (height, width) dimensions of blocks to average

    Returns:
        type: xarray.core.dataarray.DataArray

    """
    output = field.coarsen(lat=block_size[0], boundary='trim').mean()
    output = output.coarsen(lon=block_size[1], boundary='trim').mean()
    return output


# def coarsen(field, block_size):
#     def make_alpha_map(omega, amin, amax, K):
#         def alpha_map(x):
#             ys = []
#             for k in range(K):
#                 y1 = torch.sin(omega * x.sum(dim=-1) + 2 * np.pi * k / K)
#                 y2 = torch.sin(omega * (x[..., 0] - x[..., 1]) + 0.5 * np.pi * k / K)
#                 y = y1 + y2
#                 y = amax * (y - y.min()) / (y.max() - y.min()) + amin
#                 ys.append(y)
#             ys = np.stack(ys, axis=-1)
#             return ys
#         return alpha_map
#
#     field = trim(field=field, block_size=block_size)
#
#     # Create coarse lat/lon grid - conditioning variables
#     lat_lon_grid = make_lat_lon_grid_tensor(field)
#     bag_size = block_size[0] * block_size[1]
#     coarse_height = lat_lon_grid.size(0) // block_size[0]
#     coarse_width = lat_lon_grid.size(1) // block_size[1]
#     bags_lat_lon = torch.cat([torch.stack(x.split(block_size[0], dim=1))
#                               for x in lat_lon_grid.split(block_size[1])]).view(-1, bag_size, 2).mean(dim=1)
#
#     # Draw conditional weights for each bag
#     alpha_map = make_alpha_map(omega=4, amin=1e-4, amax=0.1, K=bag_size)
#     alphas = alpha_map(bags_lat_lon)
#     np.random.seed(5)
#     agg_weights = np.stack([np.random.dirichlet(alpha) for alpha in alphas])
#
#     # Aggregate based on conditional measures
#     groundtruth_tensor = torch.from_numpy(field.values)
#     gt_blocks = torch.stack([torch.stack(x.split(block_size[0], dim=1))
#                              for x in groundtruth_tensor.split(block_size[1])]).view(-1, bag_size)
#     weigthed_gt = gt_blocks * agg_weights
#     aggregate_gt = weigthed_gt.sum(dim=-1).reshape(coarse_height, coarse_width)
#
#     # Cast as xarray
#     buffer = field.coarsen(lat=block_size[0], boundary='trim').mean()
#     buffer = buffer.coarsen(lon=block_size[1], boundary='trim').mean()
#     buffer.values = aggregate_gt.numpy()
#     return buffer


def trim(field, block_size):
    """Trims dataarray to make it a multiple of the specified block size

    Args:
        field (xarray.core.dataarray.DataArray)
        block_size (tuple[int]): (height, width) dimensions of reference block

    Returns:
        type: xarray.core.dataarray.DataArray

    """
    height, width = field.shape
    trimmed_height = block_size[0] * (height // block_size[0])
    trimmed_width = block_size[1] * (width // block_size[1])
    return field[:trimmed_height, :trimmed_width]


def standardize(field):
    """Standardizes datarray

    Args:
        field (xarray.core.dataarray.DataArray)

    Returns:
        type: xarray.core.dataarray.DataArray

    """
    return (field - field.mean()) / field.std()


def make_lat_lon_grid_tensor(field):
    """Creates standardized grid torch tensor corresponding to all possible
    latitude and longitudes of the provided dataarray

    Args:
        field (xarray.core.dataarray.DataArray)

    Returns:
        type: torch.Tensor

    """
    lat = torch.from_numpy(field.lat.values)
    lon = torch.from_numpy(field.lon.values)
    lat, lon = standardize(lat), standardize(lon)
    grid = torch.stack(torch.meshgrid(lat, lon), dim=-1).float()
    return grid


def make_covariates_grid_tensor(fields):
    """Given list of registered dataarrays, stacks them together along with
    latitude/longitude grid and converts into pytorch tensor format

    Args:
        field (xarray.core.dataarray.DataArray)

    Returns:
        type: torch.Tensor

    """
    lat_long_grid = make_lat_lon_grid_tensor(fields[0])
    tensors = [torch.from_numpy(x.values).unsqueeze(-1) for x in fields]
    grid_tensor = torch.cat([lat_long_grid] + tensors, dim=-1)
    return grid_tensor


def make_bagged_dataset(covariates_grid, bags_grid, target_grid, block_size):
    """Splits fine covariate grid into blocks corresponding to coarse pixels and
    reshapes other tensors as (n_bags, ...)

    Args:
        covariates_grid (torch.Tensor): (hr_height, hr_width, ndim_covariates)
        bags_grid (torch.Tensor): (lr_height, lr_width, ndim_bags)
        target_grid (torch.Tensor): (lr_height, lr_width)
        block_size (tuple[int]): (height, width) dimensions of bags

    Returns:
        type: Description of returned object.

    """
    block_height, block_width = block_size
    bag_size = block_height * block_width

    # Split covariates grid into blocks corresponding to bags
    ndim = covariates_grid.size(-1)
    covariates_blocks = torch.cat([torch.stack(x.split(block_height, dim=1))
                                   for x in covariates_grid.split(block_width)]).view(-1, bag_size, ndim)

    # Ravel bags cells and add mean covariates values of each bag as features
    bags_blocks = torch.cat([bags_grid.view(-1, bags_grid.size(-1)),
                             covariates_blocks[..., 2:].mean(dim=1)], dim=-1)

    # Repeat bags values to match bags sizes
    extended_bags = bags_blocks.unsqueeze(1).repeat((1, bag_size, 1))

    # Flatten target grid
    targets_blocks = target_grid.view(-1)
    return covariates_blocks, bags_blocks, extended_bags, targets_blocks
