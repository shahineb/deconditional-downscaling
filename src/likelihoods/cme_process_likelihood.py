import numpy as np
import torch
from gpytorch.likelihoods import GaussianLikelihood


class CMEProcessLikelihood(GaussianLikelihood):

    def expected_log_prob(self, observations, variational_dist, cme_aggregate_covar,
                          individuals_to_cme_covar, root_inv_individuals_covar):
        """Computes expected loglikelihood under posterior variational distribution

        Args:
            observations (torch.Tensor): (n, ) tensor of aggregate observations
            variational_dist (gpytorch.distributions.MultivariateNormal): posterior
                variational distribution evaluated on joint individuals
            cme_aggregate_covar (gpytorch.lazy.LazyTensor): covariance of training bags
            individuals_to_cme_covar (torch.Tensor, gpytorch.lazy.LazyTensor)
            root_inv_individuals_covar (gpytorch.lazy.LazyTensor): square root of
                inverse of individuals covariance matrix

        Returns:
            type: torch.Tensor

        """
        # Extract variational posterior parameters
        variational_mean = variational_dist.mean
        variational_root_covar = variational_dist.lazy_covariance_matrix.root_decomposition().root

        # Store intermediate product between individuals/bags covariance and inverse root of individuals covar
        buffer = individuals_to_cme_covar.t() @ root_inv_individuals_covar

        # Compute quadratic form corresponding to obervations posterior covariance
        Q = cme_aggregate_covar.add_diag(self.noise * torch.ones(cme_aggregate_covar.shape[0])) - buffer @ buffer.t()

        # Compute logdeterminant of Q and both expectation terms
        foo, logdetQ = Q.inv_quad_logdet((buffer @ root_inv_individuals_covar.t() @ variational_root_covar).evaluate(), logdet=True)
        bar = Q.inv_quad(observations - buffer @ (root_inv_individuals_covar.t() @ variational_mean))

        # Sum up everything to obtain expected logprob under variational distribution
        output = -0.5 * (len(observations) * np.log(2 * np.pi) + logdetQ + foo + bar)
        return output