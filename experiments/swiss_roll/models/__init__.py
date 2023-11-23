from src.models import ExactGP, VariationalGP, ExactCMP, BaggedGP, VariationalCMP
from src.likelihoods import CMPLikelihood, VBaggGaussianLikelihood
from src.mlls import BagVariationalELBO
from src.utils import Registry
"""
Registry of experiment models
"""
MODELS = Registry()
TRAINERS = Registry()
PREDICTERS = Registry()


def build_model(cfg):
    model = MODELS[cfg['name']](**cfg)
    return model


def train_model(cfg):
    TRAINERS[cfg['name']](**cfg)


def predict(cfg):
    prediction = PREDICTERS[cfg['name']](**cfg)
    return prediction

from .exact_cmp_mz import *
from .exact_cmp import *
from .bagged_gp import *
from .variational_cmp import *
from .vbagg import *
from .gp_regression import *
