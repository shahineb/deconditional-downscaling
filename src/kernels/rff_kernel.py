import numpy as np
import torch
from gpytorch import kernels, lazy


class RFFKernel(kernels.RFFKernel):
    """Computes covariance matrix based on Random Fourier Features approximation

        Works for RBF and Laplace kernels

    Args:
        num_samples (int): number of random frequencies to draw
        num_dims (int): data space dimensionality, if unspecified will be inferred at
            first forward call
        which (str): kernel choice in {'rbf', 'laplace'}

    """
    __which__ = {'rbf', 'laplace'}

    def __init__(self, num_samples, num_dims=None, which='rbf', **kwargs):
        self._init_base_distribution(which=which)
        super().__init__(num_samples=num_samples, num_dims=num_dims, **kwargs)

    def _init_base_distribution(self, which):
        """Initializes frequencies samping distribution

        Args:
            which (str): kernel choice in {'rbf', 'laplace'}

        """
        if which == 'rbf':
            self.base_distribution = torch.distributions.Normal(loc=torch.zeros(1), scale=torch.ones(1))
        elif which == 'laplace':
            self.base_distribution = torch.distributions.Cauchy(loc=torch.zeros(1), scale=torch.ones(1))
        else:
            raise ValueError(f"Unknown distribution, only supports {self.__which__}")
        self.which = which

    def _init_weights(self, num_samples=None, num_dims=None, rand_weights=None):
        """Short summary.

        Args:
            num_samples (int): number of random frequencies to draw
            num_dims (int): data space dimensionality
            rand_weights (torch.Tensor): pre-sampled random frequencies

        """
        if num_dims is not None and num_samples is not None:
            d = num_dims
            D = num_samples
        if rand_weights is None:
            rand_shape = torch.Size([*self._batch_shape, d, D])
            rand_weights = self.base_distribution.sample(sample_shape=rand_shape).to(dtype=self.raw_lengthscale.dtype, device=self.raw_lengthscale.device)
        self.register_buffer("rand_weights", rand_weights)

    def _featurize(self, x, normalize=False):
        # Recompute division each time to allow backprop through lengthscale
        # Transpose lengthscale to allow for ARD
        x = x.matmul(self.rand_weights / self.lengthscale.transpose(-1, -2))
        z = torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
        if normalize:
            D = self.num_samples
            z = z / np.sqrt(D)
        return z

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **kwargs):
        if last_dim_is_batch:
            x1 = x1.transpose(-1, -2).unsqueeze(-1)
            x2 = x2.transpose(-1, -2).unsqueeze(-1)
        num_dims = x1.size(-1)
        if not hasattr(self, "rand_weights"):
            self._init_weights(num_dims, self.num_samples)
        x1_eq_x2 = torch.equal(x1, x2)
        z1 = self._featurize(x1, normalize=False)
        if not x1_eq_x2:
            z2 = self._featurize(x2, normalize=False)
        else:
            z2 = z1
        D = float(self.num_samples)
        if diag:
            return (z1 * z2).sum(-1) / D
        if x1_eq_x2:
            # Exploit low rank structure, if there are fewer features than data points
            if z1.size(-1) < z2.size(-2):
                return lazy.LowRankRootLazyTensor(z1 / np.sqrt(D))
            else:
                return lazy.RootLazyTensor(z1 / np.sqrt(D))
        else:
            return lazy.MatmulLazyTensor(z1 / D, z2.transpose(-1, -2))

    def prediction_strategy(self, train_inputs, train_prior_dist, train_labels, likelihood):
        raise NotImplementedError()
