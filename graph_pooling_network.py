import abc
from typing import List, Type
from torch_geometric.data import Data

import torch
from torch_geometric.nn import DenseGCNConv

from poolblocks.poolblock import PoolBlock


# Other example: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/proteins_diff_pool.py
class GraphPoolingNetwork(torch.nn.Module, abc.ABC):
    def __init__(self, num_node_features: int, layer_sizes: List[List[int]],
                 pool_block_args: List[dict], pooling_block_types: List[Type[PoolBlock]],
                 conv_type=Type[torch.nn.Module], activation_function=torch.nn.functional.relu,
                 forced_embeddings: float=None):
        super().__init__()
        if len(pool_block_args) != len(layer_sizes):
            raise ValueError(f"Expected the length of the pool block arguments ({len(pool_block_args)}) to be the same "
                             f"as the layer sizes ({len(layer_sizes)})!")
        if len(pooling_block_types) != len(pool_block_args):
            raise ValueError(f"Number of pooling block types ({len(pooling_block_types)}) mus be the same as the number"
                             f" of pooling block arguments provided ({len(pool_block_args)})!")
        self.layer_sizes = layer_sizes
        self.num_concepts = layer_sizes[-1][-1]
        self.pool_blocks = torch.nn.ModuleList()
        for i in range(len(pool_block_args)):
            self.pool_blocks.append(pooling_block_types[i](embedding_sizes=layer_sizes[i],
                                                           conv_type=conv_type,
                                                           activation_function=activation_function,
                                                           forced_embeddings=forced_embeddings, **pool_block_args[i]))

    @abc.abstractmethod
    def forward(self, data: Data, collect_info=False):
        pass


class DenseGraphPoolingNetwork(GraphPoolingNetwork):
    def __init__(self, num_node_features: int, layer_sizes: List[List[int]],
                 pool_block_args: List[dict], pooling_block_types: List[Type[PoolBlock]],
                 conv_type=Type[torch.nn.Module], activation_function=torch.nn.functional.relu,
                 forced_embeddings: float = None):
        super().__init__(num_node_features, layer_sizes, pool_block_args, pooling_block_types, conv_type,
                         activation_function, forced_embeddings)


    def forward(self, data: Data, collect_info=False):
        x, adj, mask = data.x, data.adj, data.mask  # to_dense_adj(data.edge_index)
        pooling_loss = 0
        if collect_info:
            pooling_assignments = []
            pooling_activations = []
            masks = []
            adjs = []
            input_embeddings = []
        else:
            pooling_assignments = pooling_activations = masks = adjs = input_embeddings = None

        for block in self.pool_blocks:
            x, adj, _, temp_loss, pool, last_embedding, mask = block(x, adj, mask)
            pooling_loss += temp_loss
            if collect_info:
                pooling_assignments.append(pool)
                pooling_activations.append(last_embedding)
                masks.append(mask)
                adjs.append(adj)
                input_embeddings.append(x)
        return x, pooling_loss, pooling_assignments, pooling_activations, mask, adjs, masks, input_embeddings

class SparseGraphPoolingNetwork(GraphPoolingNetwork):
    def __init__(self, num_node_features: int, layer_sizes: List[List[int]],
                 pool_block_args: List[dict], pooling_block_types: List[Type[PoolBlock]],
                 conv_type=Type[torch.nn.Module], activation_function=torch.nn.functional.relu,
                 forced_embeddings: float = None):
        super().__init__(num_node_features, layer_sizes, pool_block_args, pooling_block_types, conv_type,
                         activation_function, forced_embeddings)


    def forward(self, data: Data, collect_info=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        pooling_loss = 0
        edge_weights = None
        if collect_info:
            pooling_assignments = []
            pooling_activations = []
            batches = []
            edge_indices = []
            input_embeddings = []
        else:
            pooling_assignments = pooling_activations = batches = edge_indices = input_embeddings = None

        for block in self.pool_blocks:
            x, edge_index, edge_weights, temp_loss, pool, last_embedding, batch = block(x, edge_index, batch,
                                                                                        edge_weights=edge_weights)
            pooling_loss += temp_loss
            if collect_info:
                pooling_assignments.append(pool)
                pooling_activations.append(last_embedding)
                batches.append(batch)
                edge_indices.append(edge_index)
                input_embeddings.append(x)
        return x, pooling_loss, pooling_assignments, pooling_activations, batch, edge_indices, batches, input_embeddings