import abc
import typing
from argparse import Namespace

import numpy as np
import torch

from typing import List

import torch
import torch.nn.functional as F
import wandb
from torch_geometric.data import Data
from torch_geometric.nn import dense_diff_pool, DenseGCNConv

from custom_logger import log
from graphutils import adj_to_edge_index
from poolblocks.custom_asap import ASAPooling


class PoolBlock(torch.nn.Module, abc.ABC):
    def __init__(self, embedding_sizes: List[int], conv_type=DenseGCNConv,
                 activation_function=F.relu, forced_embeddings=None, **kwargs):
        super().__init__()
        self.forced_embeddings = forced_embeddings
        self.activation_function = activation_function
        self.embedding_sizes = embedding_sizes

    @abc.abstractmethod
    def forward(self, x: torch.Tensor, adj_or_edge_index: torch.Tensor, mask_or_batch=None,
                edge_weights=None):
        """
        Either takes x, adj and mask (for dense data) or x, edge_index and batch (for sparse data)
        :param x: [batch, num_nodes, num_features]
        :param adj_or_edge_index:
        :param mask_or_batch:
        :return:
        new_embeddings [batch, num_nodes_new, num_features_new] or [num_nodes_new_total, num_features_new]
        new_adj_or_edge_index
        new_edge_weights (for sparse pooling methods were necessary)
        pool_loss
        pool: assignment
        old_embeddings: Embeddings that went into the pooling operation
        batch_or_mask
        """
        pass

    def log_assignments(self, model: 'CustomNet', graphs: typing.List[Data], epoch: int):
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


    def forward(self, x: torch.Tensor, adj: torch.Tensor, mask=None, edge_weights=None):
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
        return new_embeddings, new_adj, None, loss_l + loss_e, pool, embedding, mask

    def log_assignments(self, model: 'CustomNet', graphs: typing.List[Data], epoch: int,
                        cluster_colors=torch.tensor([[1., 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [0, 0, 0]])[None, :, :]):
        node_table = wandb.Table(["graph", "pool_step", "node_index", "r", "g", "b", "label", "activations"])
        edge_table = wandb.Table(["graph", "pool_step", "source", "target"])
        device = self.embedding_convs[0].bias.device
        with torch.no_grad():
            for graph_i, data in enumerate(graphs):
                data = data.clone().detach().to(device)
                num_nodes = data.num_nodes  # note that this will be changed to tensor in model call
                out, concepts, _, pool_assignments, pool_activations = model(data)

                for pool_step, assignment in enumerate(pool_assignments[:1]):
                    # [num_nodes, num_clusters]
                    assignment = torch.softmax(assignment, dim=-1)  # usually performed by diffpool function
                    assignment = assignment.detach().cpu().squeeze(0)  # remove batch dimensions

                    if cluster_colors.shape[1] < assignment.shape[1]:
                        raise ValueError(
                            f"Only {cluster_colors.shape[1]} colors given to distinguish {assignment.shape[1]} cluster")

                    # [num_nodes, 3] (intermediate dimensions: num_nodes x num_clusters x 3)
                    colors = torch.sum(assignment[:, :, None] * cluster_colors[:, :assignment.shape[1], :], dim=1)[
                             :data.num_nodes, :]
                    colors = torch.round(colors * 255).to(int)
                    for i in range(num_nodes):
                        node_table.add_data(graph_i, pool_step, i, colors[i, 0].item(),
                                            colors[i, 1].item(), colors[i, 2].item(),
                                            ", ".join([f"{m.item() * 100:.0f}%" for m in assignment[i].cpu()]),
                                            ", ".join([f"{m.item():.2f}" for m in
                                                       pool_activations[pool_step][0, i, :].cpu()]))

                    # [3, num_edges] where the first row seems to be constant 0, indicating the graph membership
                    edge_index = adj_to_edge_index(data.adj)
                    for i in range(edge_index.shape[1]):
                        edge_table.add_data(graph_i, pool_step, edge_index[1, i].item(), edge_index[2, i].item())
        log(dict(
            # graphs_table=graphs_table
            node_table=node_table,
            edge_table=edge_table
        ), step=epoch)

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


    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch=None, edge_weights=None):

        if self.forced_embeddings is None:
            for conv in self.embedding_convs:
                x = self.activation_function(conv(x, edge_index, edge_weight=edge_weights))
        else:
            x = torch.ones(x.shape[:-1] + (self.num_output_features,), device=x.device) * self.forced_embeddings
        new_x, edge_index, new_edge_weight, batch, perm = self.asap(x=x, edge_index=edge_index, batch=batch,
                                                                    edge_weight=edge_weights)
        return new_x, edge_index, new_edge_weight, 0, perm, x, batch

    def log_assignments(self, model: 'CustomNet', graphs: typing.List[Data], epoch: int):
        node_table = wandb.Table(["graph", "pool_step", "node_index", "r", "g", "b", "label", "activations"])
        edge_table = wandb.Table(["graph", "pool_step", "source", "target"])
        device = self.embedding_convs[0].bias.device
        with torch.no_grad():
            for graph_i, data in enumerate(graphs):
                data = data.clone().detach().to(device)
                num_nodes = data.num_nodes  # note that this will be changed to tensor in model call
                out, concepts, _, pool_perms, pool_activations = model(data)

                for pool_step, perm in enumerate(pool_perms[:1]):
                    # [num_nodes]
                    perm = perm.detach().cpu()

                    # [num_nodes, 3] (intermediate dimensions: num_nodes x num_clusters x 3)
                    colors = np.ones((num_nodes, 3), dtype=np.int) * 255
                    colors[perm, :] = 0
                    for i in range(num_nodes):
                        node_table.add_data(graph_i, pool_step, i, colors[i, 0].item(),
                                            colors[i, 1].item(), colors[i, 2].item(),
                                            "",
                                            ", ".join([f"{m.item():.2f}" for m in
                                                       pool_activations[pool_step][i, :].cpu()]))

                    # [3, num_edges] where the first row seems to be constant 0, indicating the graph membership
                    edge_index = data.edge_index
                    for i in range(edge_index.shape[1]):
                        edge_table.add_data(graph_i, pool_step, edge_index[0, i].item(), edge_index[1, i].item())
        log(dict(
            # graphs_table=graphs_table
            node_table=node_table,
            edge_table=edge_table
        ), step=epoch)



__all_dense__ = [DiffPoolBlock]
__all_sparse__ = [ASAPBlock]

def from_name(name: str, dense_data: bool):
    for b in __all_dense__ if dense_data else __all_sparse__:
        if b.__name__ == name + "Block":
            return b
    raise ValueError(f"Unknown pooling type {name} for dense_data={dense_data}!")