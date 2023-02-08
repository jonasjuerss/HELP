import abc
from argparse import Namespace

import torch

from typing import List

import torch
import torch.nn.functional as F
from torch_geometric.nn import dense_diff_pool, DenseGCNConv, ASAPooling


class PoolBlock(torch.nn.Module, abc.ABC):
    def __init__(self, embedding_sizes: List[int], conv_type=DenseGCNConv,
                 activation_function=F.relu, forced_embeddings=None, **kwargs):
        super().__init__()
        self.forced_embeddings = forced_embeddings
        self.activation_function = activation_function
        self.embedding_sizes = embedding_sizes

    @abc.abstractmethod
    def forward(self, x: torch.Tensor, adj_or_edge_index: torch.Tensor, mask_or_batch=None):
        """
        Either takes x, adj and mask (for dense data) or x, edge_index and batch (for sparse data)
        :param x: [batch, num_nodes, num_features]
        :param adj_or_edge_index:
        :param mask_or_batch:
        :return:
        new_embeddings [batch, num_nodes_new, num_features_new] or [num_nodes_new_total, num_features_new]
        new_adj_or_edge_index
        pool_loss
        pool: assignment
        old_embeddings: Embeddings that went into the pooling operation
        batch_or_mask
        """
        pass


class DiffPoolBlock(PoolBlock):
    def __init__(self, embedding_sizes: List[int], conv_type=DenseGCNConv,
                 activation_function=F.relu, forced_embeddings=None, **kwargs):
        """

        :param sizes: [input_size, hidden_size1, hidden_size2, ..., output_size]
        """
        super().__init__(embedding_sizes, conv_type, activation_function, forced_embeddings)
        # Sizes of layers for generating the pooling embedding could be chosen completely arbitrary.
        # Sharing the first layers and only using a different one for the last layer would be imaginable, too.
        pool_sizes = embedding_sizes.copy()
        pool_sizes[-1] = kwargs["num_output_nodes"]

        self.embedding_convs = torch.nn.ModuleList()
        self.pool_convs = torch.nn.ModuleList()
        for i in range(len(embedding_sizes) - 1):
            # Using DenseGCNConv so I can use adjacency matrix instead of edge_index and don't have to convert back and forth for DiffPool https://github.com/pyg-team/pytorch_geometric/issues/881
            self.embedding_convs.append(conv_type(embedding_sizes[i], embedding_sizes[i+1]))
            self.pool_convs.append(conv_type(pool_sizes[i], pool_sizes[i+1]))


    def forward(self, x: torch.Tensor, adj: torch.Tensor, mask=None):
        """
        :param x:
        :param edge_index:
        :return:
        """
        embedding, pool = x, x
        if self.forced_embeddings is None:
            for conv in self.embedding_convs:
                # print("i", embedding.shape, adj.shape)
                embedding = self.activation_function(conv(embedding, adj))
                # embedding = F.dropout(embedding, training=self.training)

            # Don't need the softmax part from http://arxiv.org/abs/2207.13586 as my concepts are determined by the clusters
            # we map to. These could be found using embeddings (separately learned as here in DiffPool or just the normal
            # ones) but don't have to (like in DiffPool)
            # embedding = F.softmax(self.embedding_convs[-1](embedding, adj), dim=1)
            # max_vals, _ = torch.max(embedding, dim=1, keepdim=True)
            # embedding = embedding / max_vals
        else:
            # [batch_size, num_nodes, 1] that is 1 iff node has any neighbours
            # TODO:
            #   1. Note this also means I overwrite graphs with a single node with 0 instead of one. I suppose, I should
            #      rather use the mask
            #   2. Change feature dimension to size of final layer to make this compatible with poolblocks that change
            #      dimension (which should be basically all, wtf)
            embedding = self.forced_embeddings * torch.max(adj, dim=1)[0][:, :, None]

        for conv in self.pool_convs[:-1]:
            pool = self.activation_function(conv(pool, adj))
            # pool = F.dropout(pool, training=self.training)
        pool = self.pool_convs[-1](pool, adj)

        # TODO try dividing the softmax by its maximum value similar to the concepts
        #print(embedding.shape, edge_index.shape, pool.shape) [batch_nodes, num_features] [2, ?] []
        new_embeddings, new_adj, loss_l, loss_e = dense_diff_pool(embedding, adj, pool)
        # DiffPool will result in the same number of output nodes for all graphs, so we don't need to mask any nodes in
        # subsequent layers
        mask = None
        return new_embeddings, new_adj, loss_l + loss_e, pool, embedding, mask

class ASAPBlock(PoolBlock):
    """
    After: https://github.com/pyg-team/pytorch_geometric/blob/master/benchmark/kernel/asap.py
    """

    def __init__(self, embedding_sizes: List[int], conv_type=DenseGCNConv,
                 activation_function=F.relu, forced_embeddings=None, **kwargs):
        super().__init__(embedding_sizes, conv_type, activation_function, forced_embeddings)
        if "num_output_nodes" in kwargs:
            if "ratio_output_nodes" in kwargs:
                raise ValueError("Only a fixed number of output nodes (num_output_nodes) or a percentage of input nodes"
                                 "(ratio_output_nodes) can be defined for ASAPPooling but not both.")
            k = kwargs["num_output_nodes"]
            assert(isinstance(k, int))
        else:
            k = kwargs["ratio_output_nodes"]
            assert (isinstance(k, float))

        self.embedding_convs = torch.nn.ModuleList()
        for i in range(len(embedding_sizes) - 1):
            self.embedding_convs.append(conv_type(embedding_sizes[i], embedding_sizes[i + 1]))
        self.num_output_features = embedding_sizes[-1]

        self.asap = ASAPooling(self.num_output_features, k)


    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch=None):

        if self.forced_embeddings is None:
            for conv in self.embedding_convs:
                x = self.activation_function(conv(x, edge_index))
        else:
            x = torch.ones(x.shape[:-1] + (self.num_output_features,), device=x.device) * self.forced_embeddings
        x, edge_index, edge_weight, batch, perm = self.asap(x=x, edge_index=edge_index, batch=batch)
        return x, edge_index, 0, None, None, batch



__all_dense__ = [DiffPoolBlock]
__all_sparse__ = [ASAPBlock]

def from_name(name: str, dense_data: bool):
    for b in __all_dense__ if dense_data else __all_sparse__:
        if b.__name__ == name + "Block":
            return b
    raise ValueError(f"Unknown pooling type {name} for dense_data={dense_data}!")