from src.models import ExactCMEProcess
from src.utils import Registry
"""
Registry of experiment models
"""
MODELS = Registry()
TRAINERS = Registry()
PREDICTERS = Registry()


def build_model(**kwargs):
    model = MODELS[kwargs['name']](**kwargs)
    return model


def train_model(**kwargs):
    TRAINERS[kwargs['name']](**kwargs)


def predict(**kwargs):
    prediction = PREDICTERS[kwargs['name']](**kwargs)
    return prediction


from .exact_cme_process import *
