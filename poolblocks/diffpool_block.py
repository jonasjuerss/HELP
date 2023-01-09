from typing import List

import torch
import torch.nn.functional as F
from torch_geometric.nn import dense_diff_pool, DenseGCNConv

from custom_net import PoolBlock


class DiffPoolBlock(PoolBlock):
    def __init__(self, num_output_nodes : int, embedding_sizes : List[int], conv_type : torch.nn.Module=DenseGCNConv):
        """

        :param sizes: [input_size, hidden_size1, hidden_size2, ..., output_size]
        """
        super().__init__()
        # Sizes pf layers for generating the pooling embedding could be chosen completely arbitrary. Sharing the first layers and only using a different one for the last layer would be imaginable, too.
        pool_sizes = embedding_sizes.copy()
        pool_sizes[-1] = num_output_nodes

        self.embedding_convs = torch.nn.ModuleList()
        self.pool_convs = torch.nn.ModuleList()
        for i in range(len(embedding_sizes) - 1):
            # Using DenseGCNConv so I can use adjacency matrix instead of edge_index and don't have to convert back and forth for DiffPool https://github.com/pyg-team/pytorch_geometric/issues/881
            self.embedding_convs.append(conv_type(embedding_sizes[i], embedding_sizes[i+1]))
            self.pool_convs.append(conv_type(pool_sizes[i], pool_sizes[i+1]))


    def forward(self, x, adj, mask=None):
        """

        :param self:
        :param x:
        :param edge_index:
        :return:
        """
        embedding, pool = x, x
        for conv in self.embedding_convs[:-1]:
            # print("i", embedding.shape, adj.shape)
            embedding = F.relu(conv(embedding, adj))
            # embedding = F.dropout(embedding, training=self.training)
        embedding = F.softmax(self.embedding_convs[-1](embedding, adj), dim=1)
        max_vals, _ = torch.max(embedding, dim=1, keepdim=True)
        embedding = embedding / max_vals

        for conv in self.pool_convs[:-1]:
            pool = F.relu(conv(pool, adj))
            # pool = F.dropout(pool, training=self.training)
        pool = self.pool_convs[-1](pool, adj)

        # TODO try dividing the softmax by its maximum value similar to the concepts
        #print(embedding.shape, edge_index.shape, pool.shape) [batch_nodes, num_features] [2, ?] []
        new_embeddings, new_adj, loss_l, loss_e = dense_diff_pool(embedding, adj, pool)
        return new_embeddings, new_adj, loss_l + loss_e, pool