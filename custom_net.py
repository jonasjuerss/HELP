import abc
from typing import List

import torch
import torch.nn.functional as F
import torch_explain as explain
from torch_geometric.data import Data

class PoolBlock(torch.nn.Module, abc.ABC):
    pass

from poolblocks.diffpool_block import DiffPoolBlock

# Other example: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/proteins_diff_pool.py
class CustomNet(torch.nn.Module):
    def __init__(self, num_node_features: int, num_classes: int, layer_sizes: List[List[int]],
                 num_nodes_per_layer: List[int],
                 graph_classification = True,
                 use_entropy_layer=True,
                 pooling_block_type: PoolBlock=DiffPoolBlock, **pool_block_kwargs):
        super().__init__()
        layer_sizes[0] = [num_node_features] + layer_sizes[0]
        # assert layer_sizes[0][0] == data.num_node_features, "Number of input features must align with input features of dataset"
        for i in range(len(layer_sizes) - 1):
            assert layer_sizes[i][-1] == layer_sizes[i + 1][0], "Each block must end in the same number of features as the next one has"
        # layer_sizes[-1] = layer_sizes[-1] + [dataset.num_classes]
        #num_nodes_per_layer = [len(dataset)] + num_nodes_per_layer
        self.graph_classification = graph_classification
        if graph_classification:
            if num_nodes_per_layer[-1] != 1:
                raise ValueError("The current implementation of graph classification requires the last layer to have exactly one node")
            if use_entropy_layer:
                self.entropy_layer = explain.nn.EntropyLinear(layer_sizes[-1][-1], 1, n_classes=num_classes)
            else:
                self.entropy_layer = torch.nn.Linear(layer_sizes[-1][-1], num_classes)
        else:
            assert layer_sizes[-1][-1] == num_classes
            raise NotImplementedError("Inverse pooling for node classification not implemented yet")
        self.layer_sizes = layer_sizes
        self.use_entropy_layer = use_entropy_layer
        self.pool_blocks = torch.nn.ModuleList()
        for i in range(len(layer_sizes)):
            self.pool_blocks.append(pooling_block_type(num_nodes_per_layer[i], layer_sizes[i], **pool_block_kwargs))


    def forward(self, data: Data):
        x, adj, mask = data.x, data.adj, data.mask #to_dense_adj(data.edge_index)
        pooling_loss = 0
        pooling_assignments = []
        for block in self.pool_blocks:
            x, adj, temp_loss, pool = block(x, adj, mask)
            mask=None
            pooling_loss += temp_loss
            pooling_assignments.append(pool)

        if self.graph_classification:
            # IMPORTANT: that only works because num_nodes_per_layer[-1] == 1
            x = self.entropy_layer(x.squeeze(dim=1))
            if self.use_entropy_layer:
                x = x.squeeze(dim=2)

            return F.log_softmax(x, dim=1), pooling_loss, pooling_assignments
        else:
            raise NotImplementedError()

    def entropy_loss(self):
        return explain.nn.functional.entropy_logic_loss(self.model.entropy_layer) if self.use_entropy_layer else 0