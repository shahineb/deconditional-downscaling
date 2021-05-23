import numpy as np
import torch
from gpytorch.models import ExactGP
from gpytorch.distributions import MultivariateNormal
from gpytorch.lazy import DiagLazyTensor
from .prediction_strategies import BaggedGPPredictionStrategy


class BaggedGP(ExactGP):
    """Class implementing bagged Gaussian Process model

    Args:
        train_individuals (torch.Tensor): (N, d) tensor of individuals inputs used
            for training
        bags_sizes (list[int]): sizes of bags
        train_aggregate_targets (torch.Tensor): (n,) tensor of aggregate values
            observed for each bag
        individuals_mean (gpytorch.means.Mean): mean module used for
            individuals GP prior
        individuals_kernel (gpytorch.kernels.Kernel): covariance module
            used for individuals GP prior
        likelihood (gpytorch.likelihoods.Likelihood): observation noise likelihood model

    """
    def __init__(self, train_individuals, bags_sizes, train_aggregate_targets,
                 individuals_mean, individuals_kernel, likelihood):
        # Initialize exact GP model attributes
        super().__init__(train_inputs=train_individuals,
                         train_targets=train_aggregate_targets,
                         likelihood=likelihood)

        # Register model tensor attributes
        self.register_buffer('bags_sizes', torch.IntTensor(bags_sizes))
        self.register_buffer('train_individuals', train_individuals)
        self.register_buffer('train_aggregate_targets', train_aggregate_targets)

        # Setup model mean/kernel attributes
        self.individuals_mean = individuals_mean
        self.individuals_kernel = individuals_kernel

        # Initialize individuals posterior prediction strategy attribute
        self.individuals_prediction_strategy = None

    def _clear_cache(self):
        self.individuals_prediction_strategy = None

    def _extract_bags_evaluations_covariances(self, joint_covar, bags_sizes):
        """Extract bloc diagonals bags respective covariances from
        joint covariabce over all joint individuals, i.e.

                +-       -+
                |K^1      |
                |   .     |
            K = |     .   |    --->  [K^1, ..., K^n]
                |      K^n|
                +-       -+

        Args:
            joint_covar (gpytorch.LazyTensor)
            bags_sizes (list[int]): list of number of individual per bags

        Returns:
            type: list[torch.Tensor]
        """
        cumulative_bags_sizes = np.cumsum([0] + bags_sizes)
        bags_evaluations_covars = [joint_covar[i:j, i:j] for (i, j) in zip(cumulative_bags_sizes[:-1], cumulative_bags_sizes[1:])]
        return bags_evaluations_covars

    def _compute_bags_covariances(self, joint_covar, bags_sizes):
        """Compute diagonal matrix of bags covariance given joint covariance on bags individuals

        Args:
            joint_covar (gpytorch.LazyTensor)
            bags_sizes (list[int]): list of number of individual per bags


        Returns:
            type: (gpytorch.DiagLazyTensor)

        """
        # Extract bloc diagonal covariance matrix of each bag and aggregate
        bags_covars = self._extract_bags_evaluations_covariances(joint_covar, bags_sizes)
        diag = [torch.sum(block @ torch.ones(block.size(1), device=block.device)) / block.numel() for block in bags_covars]
        aggregate_covar = DiagLazyTensor(diag=torch.stack(diag))
        return aggregate_covar

    def forward(self, inputs, bags_sizes):
        """Aggregate Bag GP Prior computation

        Args:
            inputs (torch.Tensor): input individuals
            bags_sizes (list[int]): list of number of individual per bags

        Returns:
            type: gpytorch.distributions.MultivariateNormal

        """
        # Compute mean vector and covariance matrix on input samples
        mean = self.individuals_mean(inputs)
        covar = self.individuals_kernel(inputs)

        # Split mean vector by bags and aggregate
        mean_by_bag = mean.split(bags_sizes.squeeze().tolist())
        aggregate_mean = torch.stack([x.mean() for x in mean_by_bag])

        # Extract bloc diagonal covariance matrix of each bag and aggregate
        aggregate_covar = self._compute_bags_covariances(covar, bags_sizes.squeeze().tolist())

        # Build multivariate normal distribution of model evaluated on input samples
        prior_distribution = MultivariateNormal(mean=aggregate_mean,
                                                covariance_matrix=aggregate_covar)
        return prior_distribution

    def setup_individuals_prediction_strategy(self):
        """Defines computational strategy for deriving predictive posterior on individuals
        """
        train_aggregate_prior_dist = self.forward(self.train_individuals, self.bags_sizes)
        individuals_prediction_strategy_kwargs = {'train_individuals': self.train_individuals,
                                                  'bags_sizes': self.bags_sizes,
                                                  'train_aggregate_prior_dist': train_aggregate_prior_dist,
                                                  'train_aggregate_targets': self.train_aggregate_targets,
                                                  'likelihood': self.likelihood}
        self.individuals_prediction_strategy = BaggedGPPredictionStrategy(**individuals_prediction_strategy_kwargs)

    def get_individuals_to_aggregate_covar(self, input_individuals, input_bags_sizes):
        """Computes covariance between latent individuals GP evaluated on input
            and aggregate process GP distribution on train data

        Args:
            individuals (torch.Tensor): input individuals
            input_bags_sizes (list[int]): bags sizes of input individuals

        Returns:
            type: torch.Tensor

        """
        # Compute covariance between all individuals
        joint_covar = self.individuals_kernel(input_individuals, self.train_individuals)

        # Compute cumulative bags sizes of training and input idnividuals
        cumulative_train_bags_sizes = np.cumsum([0] + self.bags_sizes.tolist())
        cumulative_input_bags_sizes = np.cumsum([0] + input_bags_sizes)

        # Initialize cross-covariance as empty zero matrix
        individuals_to_aggregate_covar = torch.zeros(joint_covar.shape[0], len(self.bags_sizes), device=joint_covar.device)

        # If input is exactly train individuals, linear iteration on diagonal terms
        if input_individuals.size() == self.train_individuals.size():
            if torch.allclose(input_individuals, self.train_individuals):
                for idx, (i, j) in enumerate(zip(cumulative_train_bags_sizes[:-1], cumulative_train_bags_sizes[1:])):
                    bag_covar = joint_covar[i:j, i:j] @ torch.ones(j - i, device=joint_covar.device) / (j - i)
                    individuals_to_aggregate_covar[i:j, idx] = bag_covar
        # Else requires comparing bags since bags are independent
        else:
            for idx, (train_i, train_j) in enumerate(zip(cumulative_train_bags_sizes[:-1], cumulative_train_bags_sizes[1:])):
                for (input_i, input_j) in zip(cumulative_input_bags_sizes[:-1], cumulative_input_bags_sizes[1:]):
                    if train_j - train_i == input_j - input_i:
                        if torch.allclose(self.train_individuals[train_i:train_j], input_individuals[input_i:input_j]):
                            bag_covar = joint_covar[input_i:input_j, train_i:train_j] @ torch.ones(train_j - train_i, device=joint_covar.device) / (train_j - train_i)
                            individuals_to_aggregate_covar[input_i:input_j, idx] = bag_covar
                            break
        return individuals_to_aggregate_covar

    def predict(self, individuals, input_bags_sizes):
        """Run prediction of individuals posterior distribution

        Args:
            individuals (torch.Tensor): input individuals to compute posterior on
            input_bags_sizes (list[int]): bags sizes of input individuals

        Returns:
            type: gpytorch.distributions.MultivariateNormal

        """
        if self.individuals_prediction_strategy is None:
            self.setup_individuals_prediction_strategy()

        # Compute underlying GP prior mean and covariance on input
        individuals_mean = self.individuals_mean(individuals)
        individuals_covar = self.individuals_kernel(individuals)

        # Compute covariance of latent individual with aggregate process
        individuals_to_aggregate_covar = self.get_individuals_to_aggregate_covar(individuals, input_bags_sizes)

        # Compute predictive mean and covariance
        individuals_posterior_mean = self.individuals_prediction_strategy.exact_predictive_mean(individuals_mean=individuals_mean,
                                                                                                individuals_to_aggregate_covar=individuals_to_aggregate_covar)
        individuals_posterior_covar = self.individuals_prediction_strategy.exact_predictive_covar(individuals_covar=individuals_covar,
                                                                                                  individuals_to_aggregate_covar=individuals_to_aggregate_covar)

        # Encapsulate as gaussian vector
        individuals_posterior = MultivariateNormal(mean=individuals_posterior_mean,
                                                   covariance_matrix=individuals_posterior_covar)
        return individuals_posterior
