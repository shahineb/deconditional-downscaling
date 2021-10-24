from .cmp_prediction_strategy import ExactCMPPredictionStrategy
from .bagged_gp_prediction_strategy import BaggedGPPredictionStrategy

# TODO : factorize prediction strategies out of likelihoods and __init__

__all__ = ['ExactCMPPredictionStrategy', 'BaggedGPPredictionStrategy']
