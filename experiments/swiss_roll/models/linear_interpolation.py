from scipy.interpolate import interp1d
import torch
from gpytorch.distributions import MultivariateNormal
from models import MODELS, TRAINERS, PREDICTERS


@MODELS.register('linear_interpolation')
def build_swiss_roll_linear_interpolator(bags_values, aggregate_targets, **kwargs):
    """Build height-wise linear interpolator based on bags values and aggregate targets

    Args:
        bags_values (torch.Tensor)
        aggregate_targets (torch.Tensor)

    Returns:
        type: scipy.interpolate.interpolate.interp1d

    """
    model = interp1d(x=bags_values[:, -1], y=aggregate_targets, fill_value='extrapolate')
    return model


@TRAINERS.register('linear_interpolation')
def train_swiss_roll_linear_interpolator(**kwargs):
    """No training required
    """
    pass


@PREDICTERS.register('linear_interpolation')
def predict_swiss_roll_linear_interpolator(model, individuals, **kwargs):
    """Applies linear interpolator to individuals

    Args:
        model (scipy.interpolate.interpolate.interp1d)
        individuals (torch.Tensor)

    Returns:
        type: gpytorch.distributions.MultivariateNormal

    """
    individuals_pred = torch.from_numpy(model(individuals[:, -1]))
    dummy_covariance = torch.eye(len(individuals_pred))
    individuals_posterior = MultivariateNormal(mean=individuals_pred, covariance_matrix=dummy_covariance)
    return individuals_posterior
