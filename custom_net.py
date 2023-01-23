import abc
from argparse import Namespace
from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from graph_pooling_network import GraphPoolingNetwork


# Other example: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/proteins_diff_pool.py
class CustomNet(torch.nn.Module, abc.ABC):
    def __init__(self, num_node_features: int, num_classes: int, args: Namespace, device, output_layer_type,
                 pooling_block_type, conv_type):
        super().__init__()
        layer_sizes = args.layer_sizes
        num_nodes_per_layer = args.nodes_per_layer
        self.device = device
        layer_sizes[0] = [num_node_features] + layer_sizes[0]
        # assert layer_sizes[0][0] == data.num_node_features, "Number of input features must align with input features of dataset"
        for i in range(len(layer_sizes) - 1):
            assert layer_sizes[i][-1] == layer_sizes[i + 1][0],\
                "Each block must end in the same number of features as the next one has"
        self.graph_network = GraphPoolingNetwork(num_node_features, layer_sizes, num_nodes_per_layer,
                                                 pooling_block_type, conv_type)
        self.output_layer = output_layer_type(num_nodes_per_layer[-1], layer_sizes[-1][-1], num_classes, device, args)

    def custom_losses(self, batch_size: int) -> Tensor:
        """

        :param epoch: The current epoch (for wandb logging)
        :return: The weighted (sum of) loss(es) to add. Each of them can also be logged to wandb.
        """
        return self.output_layer.custom_losses(batch_size)

    def log_custom_losses(self, mode: str, epoch: int, dataset_length: int):
        """
        Logs accumulated losses and resets them to 0
        :param mode: "test" or "train"
        :param epoch:
        :param dataset_length:
        :return:
        """
        self.output_layer.log_custom_losses(mode, epoch, dataset_length)

    def forward(self, data: Data):
        concepts, pooling_loss, pooling_assignments = self.graph_network(data)
        # IMPORTANT: that only works because num_nodes_per_layer[-1] == 1
        x = self.output_layer(concepts)
        return F.log_softmax(x, dim=1), pooling_loss, pooling_assignments

    def explain(self, train_loader: DataLoader, test_loader, class_names: List[str]):
        return self.output_layer.explain(self.graph_network, train_loader, test_loader, class_names)
