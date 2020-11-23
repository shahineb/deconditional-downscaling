import pytest
import numpy as np
import torch
from gpytorch import distributions
from src.likelihoods import GaussianBagLikelihood


@pytest.fixture(scope='module')
def likelihood():
    bags_sizes = torch.IntTensor([294, 141, 128, 141, 294])
    return GaussianBagLikelihood(bags_sizes=bags_sizes)


def test_dummy_likelihood_call(likelihood):
    with torch.no_grad():
        # Compute likelihood on dummy random bag individuals evaluations
        n_individuals = likelihood.bags_sizes.sum()
        dummy_individual_evaluations = torch.randn(n_individuals)
        bagged_individuals_evaluations = dummy_individual_evaluations.split(likelihood.bags_sizes.tolist())
        dummy_likelihood = likelihood(dummy_individual_evaluations)

    # Make sure likelihood distribution mean is just bag mean aggregation of individual evaluations
    aggregated_individuals_evaluations = torch.stack([x.mean() for x in bagged_individuals_evaluations])
    assert torch.equal(dummy_likelihood.mean, aggregated_individuals_evaluations)

    # Make sure likelihood distribution covariance is just diag(σ^2/N_i)
    expected_noise_diag = likelihood.noise * torch.ones_like(likelihood.bags_sizes).div(likelihood.bags_sizes.float())
    assert torch.allclose(dummy_likelihood.scale, expected_noise_diag.sqrt(), atol=np.finfo(np.float16).eps)


def test_dummy_expected_log_prob_call(likelihood):
    # Define dummy variational distribution and observations to run actual method call
    n_individuals = likelihood.bags_sizes.sum()
    variational_mean = torch.zeros(n_individuals)
    variational_cov = torch.eye(n_individuals)
    variational_dist = distributions.MultivariateNormal(mean=variational_mean,
                                                        covariance_matrix=variational_cov)
    observations = torch.randn(len(likelihood.bags_sizes))
    with torch.no_grad():
        dummy_expected_logprob = likelihood.expected_log_prob(observations, variational_dist)

    # Build up the result we would expect with these dummy inputs
    with torch.no_grad():
        foo = observations.pow(2).mul(likelihood.bags_sizes).add(1).div(likelihood.noise).sum()
        bar = torch.log(2 * np.pi * likelihood.noise.div(likelihood.bags_sizes)).sum()
        expected_output = -0.5 * (foo + bar)

    # Make sure we get the same output
    assert np.abs(expected_output.item() - dummy_expected_logprob) < np.finfo(np.float16).eps
