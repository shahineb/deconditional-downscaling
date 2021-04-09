import gpytorch


class ExactGP(gpytorch.models.ExactGP):
    """Simplest exact GP regression model

    Args:
        mean_module (gpytorch.means.Mean): Description of parameter `mean_module`.
        covar_module (gpytorch.kernels.Kernel): Description of parameter `covar_module`.
        train_x (torch.Tensor): (n, d) tensor of training inputs
        train_y (torch.Tensor): (n, ) tensor of training targets
        likelihood (gpytorch.likelihoods.Likelihood): observation noise likelihood model

    """
    def __init__(self, mean_module, covar_module, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = mean_module
        self.covar_module = covar_module

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
