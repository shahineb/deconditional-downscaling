from gpytorch import distributions
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from .cme_process import CMEProcess


class VariationalCMEProcess(ApproximateGP, CMEProcess):
    """Approximate variational GP with inducing points module

    Args:
        landmark_points (torch.Tensor): tensor of landmark points from which to
            compute inducing points
        mean_module (gpytorch.means.Mean): mean module to compute mean vectors on inputs samples
        covar_module (gpytorch.kernels.Kernel): kernel module to compute covar matrix on input samples
    """
    def __init__(self, landmark_points, train_individuals, train_bags, train_aggregate_targets,
                 individuals_mean, individuals_kernel, bag_kernel, bags_sizes, lbda):

        # Initialize variational strategy
        variational_strategy = self._set_variational_strategy(landmark_points)
        super().__init__(variational_strategy=variational_strategy)

        # Initialize CME model parameters
        self._init_model_parameters(train_individuals=train_individuals,
                                    train_bags=train_bags,
                                    train_aggregate_targets=train_aggregate_targets,
                                    individuals_mean=individuals_mean,
                                    individuals_kernel=individuals_kernel,
                                    bag_kernel=bag_kernel,
                                    bags_sizes=bags_sizes,
                                    lbda=lbda)

    def _set_variational_strategy(self, landmark_points):
        """Sets variational family of distribution to use and variational approximation
            strategy module

        Args:
            landmark_points (torch.Tensor): tensor of landmark points from which to
                compute inducing points
        Returns:
            type: gpytorch.variational.VariationalStrategy

        """
        # Use gaussian variational family
        variational_distribution = CholeskyVariationalDistribution(num_inducing_points=landmark_points.size(0))

        # Set default variational approximation strategy
        variational_strategy = VariationalStrategy(model=self,
                                                   inducing_points=landmark_points,
                                                   variational_distribution=variational_distribution,
                                                   learn_inducing_locations=True)
        return variational_strategy

    def forward(self, inputs):
        """Defines prior distribution on input x as multivariate normal N(m(x), k(x, x))

        Args:
            inputs (torch.Tensor): input values

        Returns:
            type: gpytorch.distributions.MultivariateNormal

        """
        # Compute mean vector and covariance matrix on input samples
        mean = self.individuals_mean(inputs)
        covar = self.individuals_kernel(inputs)

        # Build multivariate normal distribution of model evaluated on input samples
        prior_distribution = distributions.MultivariateNormal(mean=mean,
                                                              covariance_matrix=covar)
        return prior_distribution

    def get_elbo_computation_parameters(self):
        """Computes tensors required to derive expected logprob term in elbo loss

        TODO : covariance matrix of individuals computed twice for individuals_to_cme_covar
        and root_inv_individuals_covar, could probably flesh it out to compute it
        once only if necessary, for now this is cleaner and poses no computational issue

        Returns:
            type: gpytorch.lazy.LazyTensor, gpytorch.lazy.LazyTensor, gpytorch.lazy.LazyTensor

        """
        cme_aggregate_covar = self.covar_module(self.train_bags)
        individuals_to_cme_covar = self.get_individuals_to_cme_covar(self.train_individuals)
        root_inv_individuals_covar = self.individuals_kernel(self.train_individuals).root_inv_decomposition().root
        return cme_aggregate_covar, individuals_to_cme_covar, root_inv_individuals_covar
