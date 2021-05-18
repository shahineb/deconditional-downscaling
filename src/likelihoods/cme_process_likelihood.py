import numpy as np
import torch
from gpytorch import lazy
from gpytorch.likelihoods import GaussianLikelihood


class CMEProcessLikelihood(GaussianLikelihood):

    def __init__(self, use_individuals_noise=True, **kwargs):
        super().__init__(**kwargs)
        self.use_individuals_noise = use_individuals_noise

    def expected_log_prob(self, observations, variational_dist, root_inv_extended_bags_covar, bags_to_extended_bags_covar, individuals_noise=None):
        """Computes expected loglikelihood under posterior variational distribution with noise applied to individuals

        Args:
            observations (torch.Tensor): (n, ) tensor of aggregate observations
            variational_dist (gpytorch.distributions.MultivariateNormal): posterior
                variational distribution evaluated on joint individuals
            root_inv_extended_bags_covar (gpytorch.lazy.LazyTensor): (L + NλI)^{-1/2}
            bags_to_extended_bags_covar (gpytorch.lazy.LazyTensor): l(y, extended_y)
            individuals_noise (torch.Tensor): σ_indiv^2

        Returns:
            type: torch.Tensor

        """
        if self.use_individuals_noise:
            output = self._compute_expected_log_prob_with_individuals_noise(observations=observations,
                                                                            variational_dist=variational_dist,
                                                                            root_inv_extended_bags_covar=root_inv_extended_bags_covar,
                                                                            bags_to_extended_bags_covar=bags_to_extended_bags_covar,
                                                                            individuals_noise=individuals_noise)
        else:
            output = self._compute_expected_log_prob_without_individuals_noise(observations=observations,
                                                                               variational_dist=variational_dist,
                                                                               root_inv_extended_bags_covar=root_inv_extended_bags_covar,
                                                                               bags_to_extended_bags_covar=bags_to_extended_bags_covar)
        return output

    def _compute_expected_log_prob_with_individuals_noise(self, observations, variational_dist, root_inv_extended_bags_covar, bags_to_extended_bags_covar, individuals_noise):
        """Computes expected loglikelihood under posterior variational distribution with noise applied to individuals

        Args:
            observations (torch.Tensor): (n, ) tensor of aggregate observations
            variational_dist (gpytorch.distributions.MultivariateNormal): posterior
                variational distribution evaluated on joint individuals
            root_inv_extended_bags_covar (gpytorch.lazy.LazyTensor): (L + NλI)^{-1/2}
            bags_to_extended_bags_covar (gpytorch.lazy.LazyTensor): l(y, extended_y)
            individuals_noise (torch.Tensor): σ_indiv^2

        Returns:
            type: torch.Tensor

        """
        # Extract variational posterior parameters
        variational_mean = variational_dist.mean
        variational_covar_cholesky = variational_dist.lazy_covariance_matrix.cholesky()

        # Compute low rank A^T aggregation term, agg_term = l(bags, extended_bags)(L + λNI)^{-1/2}
        agg_term = bags_to_extended_bags_covar @ root_inv_extended_bags_covar

        # Compute likelihood covariance matrix C = σ_indiv^2 * A^TA + σ_agg^2 I_n
        buffer = root_inv_extended_bags_covar.t() @ root_inv_extended_bags_covar
        ATA = agg_term @ buffer @ agg_term.t()
        C = ATA.mul(individuals_noise).add_diag(self.noise * torch.ones_like(observations))

        # Compute mean likelihood term (z - A^Tμ)^T C^{-1}(z - A^Tμ)
        mean_term = C.inv_quad(observations - agg_term @ (root_inv_extended_bags_covar.t() @ variational_mean))

        # Compute covariance likelihood term tr(Σpost^{1/2}T A C^{-1} A^T Σpost^{1/2})
        buffer = agg_term @ (root_inv_extended_bags_covar.t() @ variational_covar_cholesky)
        covar_term, logdetC = C.inv_quad_logdet(buffer.evaluate(), logdet=True)

        # Sum up everything to obtain expected logprob under variational distribution
        constant_term = len(observations) * np.log(2 * np.pi)
        output = -0.5 * (constant_term + logdetC + mean_term + covar_term)
        return output

    def _compute_expected_log_prob_without_individuals_noise(self, observations, variational_dist, root_inv_extended_bags_covar, bags_to_extended_bags_covar):
        """Computes expected loglikelihood under posterior variational distribution without noise applied to individuals
            Separate method from version with individuals noise as this is much more efficient

        Args:
            observations (torch.Tensor): (n, ) tensor of aggregate observations
            variational_dist (gpytorch.distributions.MultivariateNormal): posterior
                variational distribution evaluated on joint individuals
            root_inv_extended_bags_covar (gpytorch.lazy.LazyTensor): (L + NλI)^{-1/2}
            bags_to_extended_bags_covar (gpytorch.lazy.LazyTensor): l(y, extended_y)

        Returns:
            type: torch.Tensor

        """
        # Extract variational posterior parameters
        variational_mean = variational_dist.mean
        # variational_covar_cholesky = variational_dist.lazy_covariance_matrix.cholesky()
        # variational_covar = variational_dist.lazy_covariance_matrix
        variational_covar = lazy.DiagLazyTensor(diag=variational_dist.variance)

        # Compute low rank A^T aggregation term, agg_term = l(bags, extended_bags)(L + λNI)^{-1/2}
        agg_term = bags_to_extended_bags_covar @ root_inv_extended_bags_covar

        # Compute mean loss term
        mean_error = observations - agg_term @ root_inv_extended_bags_covar.t() @ variational_mean
        mean_term = torch.dot(mean_error, mean_error)

        # Compute covariance loss term
        A = root_inv_extended_bags_covar @ agg_term.t()
        foo = A.t() @ variational_covar @ A
        covar_term = foo.evaluate().diag().sum()

        # Sum up everything to obtain expected logprob under variational distribution
        constant_term = len(observations) * torch.log(2 * np.pi * self.noise)
        output = -0.5 * (constant_term + (mean_term + covar_term) / self.noise)
        return output
