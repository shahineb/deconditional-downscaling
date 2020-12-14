import torch
from gpytorch.models import ExactGP
from gpytorch.distributions import MultivariateNormal
from src.means import CMEAggregateMean
from src.kernels import CMEAggregateKernel
from .cme_prediction_strategy import ExactCMEPredictionStrategy


class ExactCMEProcess(ExactGP):
    """Class implementing exact formulation of CME Process, suitable for small datasets

    Args:
        train_individuals (torch.Tensor): (N, d) tensor of individuals inputs used
            for training
        train_bags (torch.Tensor): (n, r) tensor of bags values used for training
        train_aggregate_targets (torch.Tensor): (n,) tensor of aggregate values
            observed for each bag
        individuals_mean (gpytorch.means.Mean): mean module used for
            individuals GP prior
        individuals_kernel (gpytorch.kernels.Kernel): covariance module
            used for individuals GP prior
        bag_kernel (gpytorch.kernels.Kernel): kernel module used for bag values
        bags_sizes (list[int]): sizes of bags used
        lbda (float): inversion regularization parameter
        likelihood (gpytorch.likelihoods.Likelihood): observation noise likelihood model

    """
    def __init__(self, train_individuals, train_bags, train_aggregate_targets,
                 individuals_mean, individuals_kernel,
                 bag_kernel, bags_sizes, lbda, likelihood):
        super().__init__(train_inputs=train_bags,
                         train_targets=train_aggregate_targets,
                         likelihood=likelihood)
        self.train_individuals = train_individuals
        self.individuals_mean = individuals_mean
        self.individuals_kernel = individuals_kernel
        self.bag_kernel = bag_kernel
        self.bags_sizes = bags_sizes
        self.lbda = lbda
        self.extended_train_bags = torch.cat([torch.ones(bag_size) * bag_value
                                              for (bag_size, bag_value) in zip(bags_sizes, train_bags)])

        # Evaluate tensors needed to compute CME estimate
        latent_individuals_mean, latent_individuals_covar, root_inv_bags_covar = self._get_cme_estimate_parameters()

        # Initialize CME aggregate mean and covariance functions
        mean_module_kwargs = {'bag_kernel': self.bag_kernel,
                              'bags_values': self.extended_train_bags,
                              'individuals_mean': latent_individuals_mean,
                              'root_inv_bags_covar': root_inv_bags_covar}
        self.mean_module = CMEAggregateMean(**mean_module_kwargs)

        covar_module_kwargs = {'bag_kernel': self.bag_kernel,
                               'bags_values': self.extended_train_bags,
                               'individuals_covar': latent_individuals_covar,
                               'root_inv_bags_covar': root_inv_bags_covar}
        self.covar_module = CMEAggregateKernel(**covar_module_kwargs)

        # Initialize individuals posterior prediction strategy attribute
        self.individuals_prediction_strategy = None

    def setup_individuals_prediction_strategy(self):
        train_aggregate_prior_dist = self.forward(self.train_bags)
        individuals_prediction_strategy_kwargs = {'train_individuals': self.train_individuals,
                                                  'extended_train_bags': self.extended_train_bags,
                                                  'train_aggregate_prior_dist': train_aggregate_prior_dist,
                                                  'train_aggregate_targets': self.train_aggregate_targets,
                                                  'likelihood': self.likelihood}
        self.individuals_prediction_strategy = ExactCMEPredictionStrategy(**individuals_prediction_strategy_kwargs)

    def _get_cme_estimate_parameters(self):
        """Computes tensors required to get an estimation of the CME

        Returns:
            type: torch.Tensor, gpytorch.lazy.LazyTensor, gpytorch.lazy.LazyTensor

        """
        # Evaluate underlying GP mean and covariance on individuals
        latent_individuals_mean = self.individuals_mean(self.train_individuals)
        latent_individuals_covar = self.individuals_kernel(self.train_individuals)

        # Compute precision matrix of bags values
        bags_covar = self.bag_kernel(self.extended_train_bags)
        foo = bags_covar.add_diag(self.lbda * len(self.extended_train_bags) * torch.ones_like(self.extended_train_bags))
        root_inv_bags_covar = foo.root_inv_decomposition().root
        return latent_individuals_mean, latent_individuals_covar, root_inv_bags_covar

    def update_cme_estimate_parameters(self):
        """Update values of parameters used for CME estimate in mean and
            covariance modules

        """
        latent_individuals_mean, latent_individuals_covar, root_inv_bags_covar = self._get_cme_estimate_parameters()
        self.mean_module.individuals_mean = latent_individuals_mean
        self.mean_module.root_inv_bags_covar = root_inv_bags_covar
        self.covar_module.individuals_covar = latent_individuals_covar
        self.covar_module.root_inv_bags_covar = root_inv_bags_covar

    def forward(self, inputs):
        """CME Process Prior computation

        Args:
            inputs (torch.Tensor): input individuals

        Returns:
            type: gpytorch.distributions.MultivariateNormal

        """
        # Compute mean vector and covariance matrix on input samples
        mean = self.mean_module(inputs)
        covar = self.covar_module(inputs)

        # Build multivariate normal distribution of model evaluated on input samples
        prior_distribution = MultivariateNormal(mean=mean,
                                                covariance_matrix=covar)
        return prior_distribution

    def _get_individuals_to_aggregate_covar(self, individuals):
        """Computes covariance between latent individuals GP evaluated on input
            and CME aggregate process GP distribution on train data

        Args:
            individuals (torch.Tensor): input individuals

        Returns:
            type: torch.Tensor

        """
        individuals_covar_map = self.individuals_kernel(individuals, self.train_individuals)
        bags_covar = self.bag_kernel(self.train_bags, self.extended_train_bags)

        foo = individuals_covar_map.matmul(self.covar_module.root_inv_bags_covar)
        bar = bags_covar.matmul(self.covar_module.root_inv_bags_covar)
        output = foo.matmul(bar.t())
        return output

    def predict(self, individuals):
        """Run prediction of individuals posterior distribution

        Args:
            individuals (torch.Tensor): input individuals to compute posterior on

        Returns:
            type: gpytorch.distributions.MultivariateNormal

        """
        if self.individuals_prediction_strategy is None:
            self.setup_individuals_prediction_strategy()

        # Compute underlying GP prior mean and covariance on input
        individuals_mean = self.individuals_mean(individuals)
        individuals_covar = self.individuals_kernel(individuals)

        # Compute covariance of latent individual with CME aggregate process
        individuals_to_aggregate_covar = self._get_individuals_to_aggregate_covar(individuals)

        # Compute predictive mean and covariance
        individuals_posterior_mean = self.individuals_prediction_strategy.exact_predictive_mean(individuals_mean=individuals_mean,
                                                                                                individuals_to_train_bags_covar=individuals_to_aggregate_covar)
        individuals_posterior_covar = self.individuals_prediction_strategy.exact_predictive_covar(individuals_covar=individuals_covar,
                                                                                                  individuals_to_train_bags_covar=individuals_to_aggregate_covar)

        # Encapsulate as gaussian vector
        individuals_posterior = MultivariateNormal(mean=individuals_posterior_mean,
                                                   covariance_matrix=individuals_posterior_covar)
        return individuals_posterior

    @property
    def train_bags(self):
        return self.train_inputs[0]

    @property
    def train_aggregate_targets(self):
        return self.train_targets
