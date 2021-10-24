from abc import ABC
import torch
from gpytorch.kernels import ScaleKernel
from src.means import CMEAggregateMean
from src.kernels import CMEAggregateKernel, DeltaKernel


class CMP(ABC):
    """General class interface methods common to variations of CMP"""

    def _init_noise_kernel(self, raw_noise):
        """Initializes individuals noise kernel

        Args:
            raw_noise (float): initialization for raw noise value
                (i.e. fed to softplus transform in gpytorch)

        """
        self.noise_kernel = ScaleKernel(base_kernel=DeltaKernel())
        self.noise_kernel.initialize(raw_outputscale=raw_noise * torch.ones(1))

    def _init_cmp_mean_covar_modules(self, individuals, extended_bags_values):
        """Initializes CME aggregate mean and covariance modules based on provided
            individuals and bags values tensors

        Args:
            individuals (torch.Tensor): (N, d) tensor of individuals inputs
            extended_bags_values (torch.Tensor): (N, r) tensor of individuals bags values
        """
        # Evaluate tensors needed to compute CME estimate
        with torch.no_grad():
            latent_individuals_mean, latent_individuals_covar, root_inv_bags_covar = self._get_cmp_estimate_parameters(individuals=individuals,
                                                                                                                       extended_bags_values=extended_bags_values)

        # Initialize CME aggregate mean and covariance functions
        mean_module_kwargs = {'bag_kernel': self.bag_kernel,
                              'bags_values': extended_bags_values,
                              'individuals_mean': latent_individuals_mean,
                              'root_inv_bags_covar': root_inv_bags_covar}
        self.mean_module = CMEAggregateMean(**mean_module_kwargs)

        covar_module_kwargs = {'bag_kernel': self.bag_kernel,
                               'bags_values': extended_bags_values,
                               'individuals_covar': latent_individuals_covar,
                               'root_inv_bags_covar': root_inv_bags_covar}
        self.covar_module = CMEAggregateKernel(**covar_module_kwargs)

    def _get_cmp_estimate_parameters(self, individuals, extended_bags_values):
        """Computes tensors required to get an estimation of the CMP

        individuals (torch.Tensor): (N, d) tensor of individuals inputs
        extended_bags_values (torch.Tensor): (N, r) tensor of individuals bags values

        Returns:
            type: torch.Tensor, gpytorch.lazy.LazyTensor, gpytorch.lazy.LazyTensor

        """
        # Evaluate underlying GP mean and covariance on individuals
        latent_individuals_mean = self.individuals_mean(individuals)
        latent_individuals_covar = self.individuals_kernel(individuals)
        if self.noise_kernel is not None:
            latent_individuals_covar = latent_individuals_covar + self.noise_kernel(individuals)

        # Compute precision matrix of bags values
        bags_covar = self.bag_kernel(extended_bags_values)
        foo = bags_covar.add_diag(self.lbda * len(extended_bags_values) * torch.ones(len(extended_bags_values), device=extended_bags_values.device))
        root_inv_bags_covar = foo.root_inv_decomposition().root
        return latent_individuals_mean, latent_individuals_covar, root_inv_bags_covar

    def update_cmp_estimate_parameters(self, individuals, extended_bags_values):
        """Update values of parameters used for CMP estimate in mean and
            covariance modules

        individuals (torch.Tensor): (N, d) tensor of individuals inputs
        extended_bags_values (torch.Tensor): (N, r) tensor of individuals bags values

        """
        latent_individuals_mean, latent_individuals_covar, root_inv_bags_covar = self._get_cmp_estimate_parameters(individuals=individuals,
                                                                                                                   extended_bags_values=extended_bags_values)
        self.mean_module.individuals_mean = self.individuals_mean(individuals)
        self.mean_module.bags_values = extended_bags_values
        self.mean_module.root_inv_bags_covar = root_inv_bags_covar
        self.covar_module.bags_values = extended_bags_values
        self.covar_module.individuals_covar = latent_individuals_covar
        self.covar_module.root_inv_bags_covar = root_inv_bags_covar

    def get_individuals_to_cmp_covar(self, input_individuals, individuals, bags_values, extended_bags_values):
        """Computes covariance between latent individuals GP evaluated on input
            and CMP GP distribution on train data

        Args:
            individuals (torch.Tensor): input individuals

        Returns:
            type: torch.Tensor

        """
        individuals_covar_map = self.individuals_kernel(input_individuals, individuals)
        bags_covar = self.bag_kernel(bags_values, extended_bags_values)

        foo = individuals_covar_map.matmul(self.covar_module.root_inv_bags_covar)
        bar = bags_covar.matmul(self.covar_module.root_inv_bags_covar)
        output = foo.matmul(bar.t())
        return output
