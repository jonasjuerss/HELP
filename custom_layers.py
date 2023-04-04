from typing import List

import torch
from torch_geometric.nn import DenseGINConv
from torch import nn


def _sequential_from_hiddens(hidden_sizes: List[int]) -> nn.Sequential:
    modules = []
    for i in range(len(hidden_sizes) - 2):
        modules.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
        modules.append(nn.LeakyReLU())
    modules.append(nn.Linear(hidden_sizes[-2], hidden_sizes[-1]))
    return nn.Sequential(*modules)


class CustomDenseGINConv(DenseGINConv):
    def __init__(self, in_channels: int, out_channels: int, hidden_sizes=[128]):
        super().__init__(_sequential_from_hiddens([in_channels] + hidden_sizes + [out_channels]))
