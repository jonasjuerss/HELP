from __future__ import annotations

import abc
from functools import partial
from typing import List, Type, Callable, Tuple
from torch_geometric.data import Data

import torch
from torch_geometric.nn import DenseGCNConv

from blackbox_backprop import BlackBoxModule
from poolblocks.poolblock import PoolBlock, MonteCarloBlock


def wrap_pool_block_list(pool_blocks: List[PoolBlock], transparency: float, merge_layer: Callable) -> \
        List[PoolBlock | DifferentiablePoolingNet]:
    res = []
    for i, block in enumerate(pool_blocks):
        if isinstance(block, MonteCarloBlock):
            res.append(DifferentiablePoolingNet(block.num_mc_samples, block.perturbation, transparency,
                                                pool_blocks[i:], merge_layer))
            break
        res.append(block)
    return res


# Other example: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/proteins_diff_pool.py
class GraphPoolingNetwork(torch.nn.Module, abc.ABC):
    def __init__(self, num_node_features: int, layer_sizes: List[List[int]],
                 pool_block_args: List[dict], pooling_block_types: List[Type[PoolBlock]],
                 conv_type: Type[torch.nn.Module], use_probability_weights: bool,
                 directed_graphs: bool, activation_function=torch.nn.functional.relu, forced_embeddings: float = None,
                 transparency: float = 1):
        super().__init__()
        if len(pool_block_args) != len(layer_sizes):
            raise ValueError(f"Expected the length of the pool block arguments ({len(pool_block_args)}) to be the same "
                             f"as the layer sizes ({len(layer_sizes)})!")
        if len(pooling_block_types) != len(pool_block_args):
            raise ValueError(f"Number of pooling block types ({len(pooling_block_types)}) mus be the same as the number"
                             f" of pooling block arguments provided ({len(pool_block_args)})!")
        self.layer_sizes = layer_sizes
        self.directed_graphs = directed_graphs
        self._pool_blocks = torch.nn.ModuleList()
        for i in range(len(pool_block_args)):
            self._pool_blocks.append(pooling_block_types[i](embedding_sizes=layer_sizes[i],
                                                            conv_type=conv_type,
                                                            activation_function=activation_function,
                                                            forced_embeddings=forced_embeddings,
                                                            directed_graphs=directed_graphs,
                                                            transparency=transparency,
                                                            **pool_block_args[i]))
        for i in range(len(pool_block_args) - 1):
            if self._pool_blocks[i].output_dim != self._pool_blocks[i + 1].input_dim:
                raise ValueError("Each block must end in the same number of features as the next one has!")
        self.num_concepts = self._pool_blocks[-1].output_dim
        self.use_probability_weights = use_probability_weights
        self.transparency = transparency

    def set_final_merge_layer(self, merge_layer: Callable):
        self._pool_blocks = torch.nn.ModuleList(wrap_pool_block_list(self._pool_blocks, self.transparency, merge_layer))

    @abc.abstractmethod
    def forward(self, data: Data, collect_info=False):
        pass

    @property
    def pool_blocks(self) -> List[PoolBlock]:
        blocks = []
        for block in self._pool_blocks:
            if block.__class__ == DifferentiablePoolingNet:
                return blocks + list(block.pool_blocks)
            blocks.append(block)
        return blocks

class DifferentiablePoolingNet(BlackBoxModule):
    def __init__(self, num_samples: int, noise_distr: torch.distributions.Distribution,
                 transparency: float, pool_blocks: List[PoolBlock], merge_layer: Callable, **kwargs):
        super().__init__(num_samples, noise_distr, 2, transparency, ["adj", "mask"], **kwargs)
        self.pool_blocks = torch.nn.ModuleList([pool_blocks[0]] + wrap_pool_block_list(pool_blocks[1:], transparency,
                                                                                       merge_layer))
        self.merge_layer = merge_layer

    def preprocess(self, x: torch.Tensor, **kwargs) -> Tuple:
        return self.pool_blocks[0].preprocess(x, **kwargs), kwargs

    def hard_fn(self, x: torch.Tensor, **kwargs) -> Tuple:
        return self.pool_blocks[0].hard_fn(x, **kwargs)

    def postprocess(self, x, adj, _, __, ___, ____, _____, ______, mask) -> Tuple:
        results = []
        for i, block in enumerate(self.pool_blocks[1:]):
            if block.__class__ == DifferentiablePoolingNet:
                assert i == len(self.pool_blocks) - 2
                final_res, following_results = block(x, adj=adj, mask=mask)
                return final_res, results + following_results
            else:
                res = block(x, adj=adj, mask=mask)
                x, adj, _, _, _, _, _, _, mask = res
        return self.merge_layer(input=x, batch_or_mask=mask), results

    def forward(self, *args, **kwargs):
        block_res, (final_res, intermediate_res) = super().forward(*args, _return_intermediate=True, **kwargs)
        return final_res, [block_res] + intermediate_res

    def log_assignments(self, *args, **kwargs):
        for block in self.pool_blocks:
            block.log_assignments(*args, **kwargs)

    def log_data(self, *args, **kwargs):
        for block in self.pool_blocks:
            block.log_data(*args, **kwargs)

    def end_epoch(self):
        for block in self.pool_blocks:
            block.end_epoch()

    def __getattr__(self, item):
        if item in ["cluster_alg", "last_num_clusters", "pooling_assignments"]:
            return getattr(self.pool_blocks[0], item)
        return super().__getattr__(item)
        # Would be cleaner but doesn't work because of the way torch.nn.Module overrides __getattr__
        # if (item.startswith('__') and item.endswith('__')):
        #     return super().__getattr__(item)
        # return getattr(self.pool_blocks[0], item)


class DenseGraphPoolingNetwork(GraphPoolingNetwork):
    def __init__(self, num_node_features: int, layer_sizes: List[List[int]],
                 pool_block_args: List[dict], pooling_block_types: List[Type[PoolBlock]],
                 conv_type: Type[torch.nn.Module], use_probability_weights: bool,
                 directed_graphs: bool, activation_function=torch.nn.functional.relu,
                 forced_embeddings: float = None, transparency: float = 1):
        super().__init__(num_node_features, layer_sizes, pool_block_args, pooling_block_types, conv_type,
                         use_probability_weights, directed_graphs, activation_function, forced_embeddings, transparency)

    def forward(self, data: Data, collect_info=False):
        x, adj, mask = data.x, data.adj, data.mask  # to_dense_adj(data.edge_index)
        pooling_loss = 0
        probabilities = None
        if collect_info:
            pooling_assignments = []
            node_assignments = []
            pooling_activations = []
            masks = []
            adjs = []
            input_embeddings = []
        else:
            pooling_assignments = node_assignments = pooling_activations = masks = adjs = input_embeddings = None

        # TODO incorporate in sparse or finally merge them
        skipped_result = None
        results = []
        for i, block in enumerate(self._pool_blocks):
            if block.__class__ == DifferentiablePoolingNet:
                assert i == len(self._pool_blocks) - 1
                skipped_res, res = block(x, adj=adj, mask=mask)
                results += res
                skipped_result = skipped_res
            else:
                res = block(x, adj=adj, mask=mask)
                x, adj, _, probs, temp_loss, pool, node_ass, last_embedding, mask = res
                results.append(res)

        for i, res in enumerate(results):
            x_tmp, adj, _, probs, temp_loss, pool, node_ass, last_embedding, mask = res
            pooling_loss += temp_loss
            if collect_info:
                pooling_assignments.append(pool)
                node_assignments.append(node_ass)
                pooling_activations.append(last_embedding)
                masks.append(mask)
                adjs.append(adj)
                input_embeddings.append(x_tmp)
            if self.use_probability_weights and probs is not None:
                probabilities = probs if probabilities is None\
                    else torch.repeat_interleave(probabilities, self._pool_blocks[i].num_mc_samples) * probs
        return x, probabilities, pooling_loss, pooling_assignments, node_assignments, pooling_activations, mask, adjs, masks, \
            input_embeddings, skipped_result


class SparseGraphPoolingNetwork(GraphPoolingNetwork):
    def __init__(self, num_node_features: int, layer_sizes: List[List[int]],
                 pool_block_args: List[dict], pooling_block_types: List[Type[PoolBlock]],
                 conv_type: Type[torch.nn.Module], use_probability_weights: bool, directed_graphs: bool,
                 activation_function=torch.nn.functional.relu, forced_embeddings: float = None,
                 transparency: float = 1):
        super().__init__(num_node_features, layer_sizes, pool_block_args, pooling_block_types, conv_type,
                         use_probability_weights, directed_graphs, activation_function, forced_embeddings, transparency)

    def forward(self, data: Data, collect_info=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        pooling_loss = 0
        probabilities = None
        edge_weights = None
        if collect_info:
            pooling_assignments = []
            node_assignments = []
            pooling_activations = []
            batches = []
            edge_indices = []
            input_embeddings = []
        else:
            pooling_assignments = node_assignments = pooling_activations = batches = edge_indices = input_embeddings\
                = None

        for block in self.pool_blocks:
            x, edge_index, edge_weights, probs, temp_loss, pool, node_ass, last_embedding, batch = block(x, edge_index,
                                                                                                         batch,
                                                                                                         edge_weights=edge_weights)
            pooling_loss += temp_loss
            if collect_info:
                pooling_assignments.append(pool)
                node_assignments.append(node_ass)
                pooling_activations.append(last_embedding)
                batches.append(batch)
                edge_indices.append(edge_index)
                input_embeddings.append(x)
            if self.use_probability_weights and probs is not None:
                probabilities = probs if probabilities is None else probabilities * probs
        return x, probabilities, pooling_loss, pooling_assignments, node_assignments, pooling_activations, batch,\
            edge_indices, batches, input_embeddings, None
