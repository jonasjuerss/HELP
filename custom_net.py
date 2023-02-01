import abc
from argparse import Namespace
from typing import List, Type

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from function_module import FunctionModule
from graph_pooling_network import GraphPoolingNetwork, DenseGraphPoolingNetwork
from poolblocks.poolblock import PoolBlock


# Other example: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/proteins_diff_pool.py
class CustomNet(torch.nn.Module, abc.ABC):
    def __init__(self, num_node_features: int, num_classes: int, args: Namespace, device, output_layer_type,
                 pooling_block_type: Type[PoolBlock], conv_type: Type[torch.nn.Module], activation_function):
        super().__init__()
        layer_sizes = args.layer_sizes
        pool_block_args: List[dict] = args.pool_block_args
        self.device = device
        layer_sizes[0] = [num_node_features] + layer_sizes[0]
        # assert layer_sizes[0][0] == data.num_node_features, "Number of input features must align with input features of dataset"
        for i in range(len(layer_sizes) - 1):
            assert layer_sizes[i][-1] == layer_sizes[i + 1][0],\
                "Each block must end in the same number of features as the next one has"
        self.graph_network = DenseGraphPoolingNetwork(num_node_features, layer_sizes, pool_block_args,
                                                      pooling_block_type, conv_type=conv_type,
                                                      activation_function=activation_function,
                                                      forced_embeddings=args.forced_embeddings)

        num_output_nodes = pool_block_args[-1].get("num_output_nodes", None)
        # Recall that input to this is: [batch_size, num_output_nodes, features_per_output_node]
        if args.output_layer_merge == "flatten":
            self.merge_layer = torch.nn.Flatten(start_dim=-2, end_dim=-1)  # merges the last two dimensions
            if num_output_nodes is None:
                raise ValueError("The flattened output of all node embeddings is only constant if the number of nodes "
                                 "in the graph produced by the last pooling step is.")
            gnn_output_shape = (layer_sizes[-1][-1] * num_output_nodes, )
        elif args.output_layer_merge == "none":
            gnn_output_shape = (num_output_nodes, layer_sizes[-1][-1])
            self.merge_layer = torch.nn.Identity()
        elif args.output_layer_merge == "sum":
            self.merge_layer = FunctionModule(torch.sum, dim=-1)
            gnn_output_shape = (layer_sizes[-1][-1], )
        elif args.output_layer_merge == "avg":
            self.merge_layer = FunctionModule(torch.mean, dim=-1)
            gnn_output_shape = (layer_sizes[-1][-1], )


        self.output_layer = output_layer_type(gnn_output_shape, num_classes, device, args)

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
        ndim = data.x.ndim
        if ndim == 2:
            data.x = data.x[None, ...]
            data.y = data.y[None, ...]
            data.mask = data.mask[None, ...]
            data.num_nodes = torch.tensor([data.num_nodes], device=data.x.device)
            if data.adj is not None:
                data.adj = data.adj[None, ...]
            if data.edge_index is not None:
                data.edge_index = data.edge_index[None, ...]
        elif ndim > 3:
            print("Multiple batch dimensions currently might not work as expected!")

        concepts, pooling_loss, pooling_assignments, pooling_activations = self.graph_network(data)

        # IMPORTANT: that only works because num_nodes_per_layer[-1] == 1
        x = self.output_layer(self.merge_layer(concepts))
        return F.log_softmax(x, dim=1), concepts, pooling_loss, pooling_assignments, pooling_activations

    def explain(self, train_loader: DataLoader, test_loader: DataLoader, class_names: List[str]):
        return self.output_layer.explain(self.graph_network, train_loader, test_loader, class_names)
