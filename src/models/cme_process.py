from abc import ABC
import torch
from src.means import CMEAggregateMean
from src.kernels import CMEAggregateKernel


class CMEProcess(ABC):
    """General class interface methods common to variations of CME process"""

    def _init_model_parameters(self, train_individuals, train_bags, train_aggregate_targets,
                               individuals_mean, individuals_kernel, bag_kernel, bags_sizes, lbda):
        """TO DO : find a way to do this in class __init__ method and use
            multiple inheritance to propagate to child classes, as of now
            conflict with first parent class

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

        """
        # Setup model attributes
        self.train_individuals = train_individuals
        self.train_bags = train_bags
        self.train_aggregate_targets = train_aggregate_targets
        self.individuals_mean = individuals_mean
        self.individuals_kernel = individuals_kernel
        self.bag_kernel = bag_kernel
        self.bags_sizes = bags_sizes
        self.lbda = lbda
        self.extended_train_bags = torch.cat([bag_value.repeat(bag_size, 1)
                                              for (bag_size, bag_value) in zip(bags_sizes, train_bags)]).squeeze()

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
        foo = bags_covar.add_diag(self.lbda * len(self.extended_train_bags) * torch.ones(len(self.extended_train_bags)))
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

    def get_individuals_to_cme_covar(self, individuals):
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
