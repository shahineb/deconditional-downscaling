import os
import sys

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../..")
sys.path.append(base_dir)

from src.models import VariationalGP, VariationalCMEProcess, VBaggGaussianLikelihood
from src.likelihoods import CMEProcessLikelihood
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


from .variational_cme_process import *
from .krigging import *
from .vbagg import *
