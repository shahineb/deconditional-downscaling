import numpy as np
import torch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch import distributions


class VBaggGaussianLikelihood(GaussianLikelihood):
    """Gaussian likelihood modelling aggregated bag observation y_a against bag
    individuals B_a under individuals regression model f

        y_a = Aggregate(f(B_a)) + ε  where ε ~ N(0, σ^2/bag_size)

    """
    def forward(self, individuals_evaluations, bags_sizes):
        """Computes aggregated observations vector y = [y_1, ..., y_n]^T likelihood under
        regression model f given regression model bags evaluations f(B) = [f(B_1), ..., f(B_n)]^T

        Assuming the n bags independence, we have

            p(y|f(B)) = Π p(y_a|f(B_a)) = Π N(Aggregate(f(B_a)), σ^2/bag_size)

        Args:
            individuals_evaluations (torch.Tensor): (Σ bags_sizes, ) tensor of bags individuals
                evaluated for regression model f(B)
            bags_sizes (list[int]): (N_bags, ) list of bags sizes

        Returns:
            type: torch.distributions.normal.Normal
        """
        # Split and aggregate f(B) into [Agg(f(B_1)), ..., Agg(f(B_n))]
        individuals_evaluations_by_bag = individuals_evaluations.split(bags_sizes)
        aggregated_bags_evaluations = torch.stack([x.mean() for x in individuals_evaluations_by_bag])

        # Get noise homoskedastic covariance diagonal vector sized and scale dimensions by bag sizes
        noise_covar = self.noise_covar(shape=aggregated_bags_evaluations.shape).diag()
        noise_covar.div_(bags_sizes)

        # Build joint normal distribution centered on aggregated bags evaluations corresponding bags noises
        likelihood_distribution = distributions.base_distributions.Normal(loc=aggregated_bags_evaluations,
                                                                          scale=noise_covar.sqrt())

        return likelihood_distribution

    def _extract_bags_evaluations_covariances(self, variational_dist, bags_sizes):
        """Extract bloc diagonals bags respective variational posterior covariances from
        variational posterior distribution over all joint individuals, i.e.

                +-       -+
                |S^1      |
                |   .     |
            S = |     .   |    --->  [S^1, ..., S^n]
                |      S^n|
                +-       -+

        Args:
            variational_dist (gpytorch.distributions.MultivariateNormal): posterior variational
                distribution evaluate on joint individuals q(f(B))
            bags_sizes (list[int]): (N_bags, ) list of number of individual per bags

        Returns:
            type: list[torch.Tensor]
        """
        cumulative_bags_sizes = np.cumsum([0] + bags_sizes)
        joint_evaluations_covar = variational_dist.covariance_matrix
        bags_evaluations_covars = [joint_evaluations_covar[i:j, i:j] for (i, j) in zip(cumulative_bags_sizes[:-1], cumulative_bags_sizes[1:])]
        return bags_evaluations_covars

    def expected_log_prob(self, observations, variational_dist, bags_sizes):
        """Compute expected loglikelihood under posterior variational distribution q

        TODO: Would probably be great to rewrite this implementation

        Args:
            observations (torch.Tensor): (N_bags, ) tensors of aggregated bags observation
                y = [y_1, ..., y_n]^T
            variational_dist (gpytorch.distributions.MultivariateNormal): posterior variational
                distribution evaluate on joint individuals q(f(B))
            bags_sizes (list[int]): list of number of individual points per bags

        Returns:
            type: torch.Tensor

        """
        # Extract bags respective mean and covariance m^a and S^a
        bags_evaluations_means = variational_dist.mean.split(bags_sizes)
        bags_evaluations_covars = self._extract_bags_evaluations_covariances(variational_dist, bags_sizes)

        # Compute log of term under exponential
        foo = 2 * observations * torch.stack([m.sum() for m in bags_evaluations_means])
        bar = torch.stack([S.add(m.ger(m)).sum() for (m, S) in zip(bags_evaluations_means,
                                                                   bags_evaluations_covars)])
        bags_sizes = torch.Tensor(bags_sizes, device=observations.device)
        first_term = observations.pow(2) - foo.div(bags_sizes) + bar.div(bags_sizes.pow(2))
        first_term.mul_(bags_sizes / self.noise)

        # Compute log of scaling term
        second_term = torch.log(2 * np.pi * self.noise / bags_sizes)

        # Sum and reduce on all bags
        output = -0.5 * torch.sum(first_term + second_term)
        return output
