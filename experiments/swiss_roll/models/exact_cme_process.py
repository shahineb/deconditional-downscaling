import torch
import gpytorch
from progress.bar import Bar
from models import ExactCMEProcess, MODELS, TRAINERS, PREDICTERS


@MODELS.register('exact_cme_process')
def build_swiss_roll_exact_cme_process(individuals, bags_values, aggregate_targets, bags_sizes, lbda, **kwargs):
    """Hard-coded initialization of Exact CME Process module used for swiss roll experiment

    Args:
        individuals (torch.Tensor)
        bags_values (torch.Tensor)
        aggregate_targets (torch.Tensor)
        bags_sizes (list[int])
        lbda (float)

    Returns:
        type: ExactCMEProcess

    """
    # Inverse softplus utility for gpytorch lengthscale intialization
    inv_softplus = lambda x, n: torch.log(torch.exp(x * torch.ones(n)) - 1)

    # Define mean and covariance modules
    individuals_mean = gpytorch.means.ZeroMean()

    # Define individuals kernel
    base_individuals_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=3)
    base_individuals_kernel.initialize(raw_lengthscale=inv_softplus(x=1, n=3))
    individuals_kernel = gpytorch.kernels.ScaleKernel(base_individuals_kernel)

    # Define bags kernels
    base_bag_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=3)
    base_bag_kernel.initialize(raw_lengthscale=inv_softplus(x=1, n=3))
    bag_kernel = gpytorch.kernels.ScaleKernel(base_bag_kernel)

    # Define model
    model = ExactCMEProcess(individuals_mean=individuals_mean,
                            individuals_kernel=individuals_kernel,
                            bag_kernel=bag_kernel,
                            train_individuals=individuals,
                            train_bags=bags_values,
                            train_aggregate_targets=aggregate_targets,
                            bags_sizes=bags_sizes,
                            lbda=lbda,
                            likelihood=gpytorch.likelihoods.GaussianLikelihood())
    return model


@TRAINERS.register('exact_cme_process')
def train_swiss_roll_exact_cme_process(model, lr, n_epochs, **kwargs):
    """Hard-coded training script of Exact CME Process for swiss roll experiment

    Args:
        model (ExactCMEProcess)
        lr (float)
        n_epochs (int)

    """
    # Set model in training mode
    model.train()
    model.likelihood.train()

    # Define optimizer and exact loglikelihood module
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    bar = Bar("Epoch", max=n_epochs)
    for _ in range(n_epochs):
        # Zero-out remaining gradients
        optimizer.zero_grad()

        # Compute marginal distribution p(.|x,y)
        output = model(model.train_bags)

        # Evaluate -logp(z|x, y) on aggregate observations z
        loss = -mll(output, model.train_aggregate_targets)

        # Take gradient step
        loss.backward()
        optimizer.step()

        # Update aggregation operators based on new hyperparameters
        with torch.no_grad():
            model.update_cme_estimate_parameters()

        bar.suffix = f"NLL {loss.item()}"
        bar.next()


@PREDICTERS.register('exact_cme_process')
def predict_swiss_roll_exact_cme_process(model, individuals, **kwargs):
    """Hard-coded prediciton of individuals posterior for Exact CME Process on
    swiss roll experiment

    Args:
        model (ExactCMEProcess)
        individuals (torch.Tensor)

    Returns:
        type: gpytorch.distributions.MultivariateNormal

    """
    # Set model in evaluation mode
    model.eval()
    model.likelihood.eval()

    # Compute predictive posterior on individuals
    with torch.no_grad():
        individuals_posterior = model.predict(individuals)
    return individuals_posterior
