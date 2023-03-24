import abc
from argparse import Namespace
from typing import List, Type

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn.aggr import MeanAggregation, SumAggregation

from function_module import FunctionModule, MaskedFlatten, MaskedMean, MaskedSum
from graph_pooling_network import GraphPoolingNetwork, DenseGraphPoolingNetwork, SparseGraphPoolingNetwork
from output_layers import Classifier
from poolblocks.poolblock import PoolBlock



# Other example: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/proteins_diff_pool.py
class CustomNet(torch.nn.Module, abc.ABC):
    def __init__(self, num_node_features: int, num_classes: int, args: Namespace, device,
                 output_layer_type: Type[Classifier], pooling_block_type: Type[PoolBlock],
                 conv_type: Type[torch.nn.Module], activation_function):
        super().__init__()
        layer_sizes = args.layer_sizes
        pool_block_args: List[dict] = args.pool_block_args
        self.device = device
        self.dense_data = args.dense_data
        layer_sizes[0] = [num_node_features] + layer_sizes[0]
        # assert layer_sizes[0][0] == data.num_node_features, "Number of input features must align with input features of dataset"
        for i in range(len(layer_sizes) - 1):
            assert layer_sizes[i][-1] == layer_sizes[i + 1][0],\
                "Each block must end in the same number of features as the next one has"

        network_type = DenseGraphPoolingNetwork if self.dense_data else SparseGraphPoolingNetwork
        self.graph_network = network_type(num_node_features, layer_sizes, pool_block_args,
                                          pooling_block_type, conv_type=conv_type,
                                          activation_function=activation_function,
                                          forced_embeddings=args.forced_embeddings)

        num_output_nodes = pool_block_args[-1].get("num_output_nodes", None)
        # Recall that for dense data:
        # x: [batch_size, max_num_output_nodes, features_per_output_node]
        # mask: [batch_size, max_num_output_nodes] (booleans)
        #
        # and for sparse data:
        # x: [num_nodes_total, num_features]
        # batch: [num_nodes_total] (numbers in [0, batch_size))
        if args.output_layer_merge == "flatten":
            if num_output_nodes is None:
                raise ValueError("The flattened output of all node embeddings is only constant if the number of "
                                 "nodes in the graph produced by the last pooling step is.")
            if self.dense_data:
                # self.merge_layer = torch.nn.Flatten(start_dim=-2, end_dim=-1)  # merges the last two dimensions
                # Just to be sure. Theoretically, if we have the same number of nodes in each graph, none of them
                # should be masked so the "normal" flatten above should be sufficient
                self.merge_layer = MaskedFlatten()
            else:
                self.merge_layer = FunctionModule(torch.reshape, {"input": "input"},
                                                  shape=(-1, layer_sizes[-1][-1] * num_output_nodes))
            gnn_output_shape = (layer_sizes[-1][-1] * num_output_nodes, )
        elif args.output_layer_merge == "none":
            # TODO rethink this when I know what input Pietro's layer wants. The current format of
            #  [num_nodes, num_features] doesn't work because it requires a fixed number of nodes. Not requiring this is
            #  the whole point of this mode.
            #  EDIT: I do allow num_output_nodes = None for a flexible dimension in poolblock but I don
            if self.dense_data:
                pass
                #self.merge_layer = torch.nn.Identity()
            else:
                pass
            gnn_output_shape = (num_output_nodes, layer_sizes[-1][-1])
        elif args.output_layer_merge == "sum":
            if self.dense_data:
                self.merge_layer = MaskedSum()
            else:
                self.merge_layer = FunctionModule(SumAggregation(), {"x": "input", "index": "batch_or_mask"}, dim=-2)
            gnn_output_shape = (layer_sizes[-1][-1], )
        elif args.output_layer_merge == "avg":
            if self.dense_data:
                self.merge_layer = MaskedMean()
            else:
                self.merge_layer = FunctionModule(MeanAggregation(), {"x": "input", "index": "batch_or_mask"}, dim=-2)
            gnn_output_shape = (layer_sizes[-1][-1], )

        if self.merge_layer is None:
            raise ValueError(f"Unsupported merge operation for {'dense' if args.dense_data else 'sparse'} data: "
                             f"{args.output_layer_merge}")

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
        """

        :param data:
        :return:
            - out: [batch_size, num_classes] log softmax predictions
            - concepts: []
            - pooling_loss: additional loss from pooling (or 0)
            - pooling_assignment:
            - pooling_activations:
            - all_batch_or_mask list of batch/mask tensors after each pooling step. Note that this does not include the
                initial one given by the data object
        """
        ndim = data.x.ndim
        # Note that a batch dimension is only needed for dense representation
        if ndim == 2 and hasattr(data, "adj"):
            data.x = data.x[None, ...]
            data.y = data.y[None, ...]
            data.num_nodes = torch.tensor([data.num_nodes], device=data.x.device)
            data.adj = data.adj[None, ...]
            if hasattr(data, "mask"):
                data.mask = data.mask[None, ...]
        elif ndim > 3:
            print("Multiple batch dimensions currently might not work as expected!")

        concepts, pooling_loss, pooling_assignments, pooling_activations, batch_or_mask, adjs_or_edge_indices,\
            all_batch_or_mask = self.graph_network(data)
        x = self.output_layer(self.merge_layer(input=concepts, batch_or_mask=batch_or_mask))
        return F.log_softmax(x, dim=1), concepts, pooling_loss, pooling_assignments, pooling_activations,\
            adjs_or_edge_indices, all_batch_or_mask

    def explain(self, train_loader: DataLoader, test_loader: DataLoader, class_names: List[str]):
        return self.output_layer.explain(self.graph_network, train_loader, test_loader, class_names)
