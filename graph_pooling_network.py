import abc
from typing import List
from torch_geometric.data import Data

import torch
from torch_geometric.nn import DenseGCNConv

from poolblocks.poolblock import DiffPoolBlock


# Other example: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/proteins_diff_pool.py
class GraphPoolingNetwork(torch.nn.Module):
    def __init__(self, num_node_features: int, layer_sizes: List[List[int]],
                 num_nodes_per_layer: List[int],
                 pooling_block_type=DiffPoolBlock,
                 conv_type=DenseGCNConv):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.num_concepts = layer_sizes[-1][-1]
        self.pool_blocks = torch.nn.ModuleList()
        for i in range(len(layer_sizes)):
            self.pool_blocks.append(pooling_block_type(num_nodes_per_layer[i], layer_sizes[i], conv_type=conv_type))


    def forward(self, data: Data):
        x, adj, mask = data.x, data.adj, data.mask  # to_dense_adj(data.edge_index)
        pooling_loss = 0
        pooling_assignments = []
        pooling_activations = []
        for block in self.pool_blocks:
            x, adj, temp_loss, pool, last_embedding = block(x, adj, mask)
            mask = None
            pooling_loss += temp_loss
            pooling_assignments.append(pool)
            pooling_activations.append(last_embedding)
        return x, pooling_loss, pooling_assignments, pooling_activations