"""
Author : Victor Fairon
Date : 2026-09-09

MLP architecture for the surface-code decoder. Pure model definition for both train.py and tune.py to
build a model purely from a config dict.
"""
from collections import OrderedDict

import torch.nn as nn

DEFAULT_CONFIG = {
    "hidden_sizes": (256, 128, 64),
    "dropout": (0.1, 0.1, 0.1),   
    "lr": 0.003,
}


def build_model(config, input_size, output_size=1):
    """
    Build the MLP decoder from a config dict.

    @args:
        config : dict with "hidden_sizes" (tuple of ints) and "dropout" (tuple of floats, one per hidden layer) keys
        input_size : number of syndrome bits fed in ((d²-1)*rounds)
        output_size : number of logical observables to predict (1 for memory_z)
    @returns:
        nn.Sequential
    """
    hidden_sizes = config["hidden_sizes"]
    dropout_p = config["dropout"]

    layers = OrderedDict()
    # the * unpacks the hidden size tuple and so basically contatenates both things
    sizes = [input_size, *hidden_sizes]
    for i in range(len(hidden_sizes)):
        #size = size of previous layer
        layers[f"fc{i+1}"] = nn.Linear(sizes[i], sizes[i + 1])
        layers[f"relu{i+1}"] = nn.ReLU()
        layers[f"drp{i+1}"] = nn.Dropout(dropout_p[i])
    layers[f"fc{len(hidden_sizes)+1}"] = nn.Linear(hidden_sizes[-1], output_size)

    return nn.Sequential(layers)