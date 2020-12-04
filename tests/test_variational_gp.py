import pytest
import numpy as np
import torch
import gpytorch
from src.models import VariationalGP


@pytest.fixture(scope='module')
def model():
    landmark_points = torch.linspace(0, 1, 50)
    model = VariationalGP(landmark_points=landmark_points,
                          mean_module=gpytorch.means.ZeroMean(),
                          covar_module=gpytorch.kernels.RBFKernel())
    return model


def test_dummy_variational_posterior_call(model):
    # Evaluate variational GP model on its own inducing points
    with torch.no_grad():
        landmark_points = model.variational_strategy.inducing_points.squeeze()
        variational_posterior = model(landmark_points)

    # Posterior and variational means should be the same
    assert torch.allclose(variational_posterior.mean,
                          model.variational_strategy.variational_distribution.mean,
                          atol=0.01)

    # Posterior covariance should be Kww - I + Σu
    expected_covariance = model.covar_module(landmark_points).evaluate() - torch.eye(len(landmark_points))
    expected_covariance = expected_covariance + model.variational_strategy.variational_distribution.covariance_matrix
    assert torch.allclose(variational_posterior.covariance_matrix,
                          expected_covariance,
                          atol=np.finfo(np.float16).eps)
