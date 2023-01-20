import abc
from typing import List

import torch
import torch.nn.functional as F
import torch_explain as explain
from torch_explain.logic import test_explanation
from torch_explain.logic.nn import entropy
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


class PoolBlock(torch.nn.Module, abc.ABC):
    pass

from poolblocks.diffpool_block import DiffPoolBlock

# Other example: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/proteins_diff_pool.py
class CustomNet(torch.nn.Module):
    def __init__(self, num_node_features: int, num_classes: int, layer_sizes: List[List[int]],
                 num_nodes_per_layer: List[int], device,
                 graph_classification = True,
                 use_entropy_layer=True,
                 pooling_block_type: PoolBlock=DiffPoolBlock,
                 **pool_block_kwargs):
        super().__init__()
        self.device = device
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
                self.dense_part = torch.nn.Sequential(
                    explain.nn.EntropyLinear(layer_sizes[-1][-1], 1, n_classes=num_classes),
                    torch.nn.Flatten(1))  # equivalent to: squeeze(2) (NOT squeeze(1))
            else:
                self.dense_part = torch.nn.Sequential(torch.nn.Linear(layer_sizes[-1][-1], num_classes))
        else:
            assert layer_sizes[-1][-1] == num_classes
            raise NotImplementedError("Inverse pooling for node classification not implemented yet")
        self.layer_sizes = layer_sizes
        self.use_entropy_layer = use_entropy_layer
        self.pool_blocks = torch.nn.ModuleList()
        for i in range(len(layer_sizes)):
            self.pool_blocks.append(pooling_block_type(num_nodes_per_layer[i], layer_sizes[i], **pool_block_kwargs))


    def graph_part(self, data: Data):
        x, adj, mask = data.x, data.adj, data.mask  # to_dense_adj(data.edge_index)
        pooling_loss = 0
        pooling_assignments = []
        for block in self.pool_blocks:
            x, adj, temp_loss, pool = block(x, adj, mask)
            mask = None
            pooling_loss += temp_loss
            pooling_assignments.append(pool)
        return x, pooling_loss, pooling_assignments

    def forward(self, data: Data):
        concepts, pooling_loss, pooling_assignments = self.graph_part(data)

        if self.graph_classification:
            # IMPORTANT: that only works because num_nodes_per_layer[-1] == 1
            x = self.dense_part(concepts.squeeze(dim=1))
            return F.log_softmax(x, dim=1), pooling_loss, pooling_assignments
        else:
            raise NotImplementedError()

    def entropy_loss(self):
        return explain.nn.functional.entropy_logic_loss(self.dense_part) if self.use_entropy_layer else 0

    def explain(self, train_loader: DataLoader, test_loader, class_names: List[str]):
        if not self.use_entropy_layer:
            return None
            #raise ValueError("Cannot explain a model that doesn't use an entropy layer!")

        # determining the size in advance would be more efficient than stacking
        # [num_samples, gnn_output_size]
        xs = torch.empty((0, self.layer_sizes[-1][-1]), device=self.device)
        # [num_samples]
        ys = torch.empty((0, ), device=self.device, dtype=torch.long)
        train_samples = 0
        for data in train_loader:
            # data.x: [batch_size, nodes_per_graph, input_feature_size/num_colors]
            # data.y: [batch_size, 1]
            concepts, _, _ = self.graph_part(data.to(self.device))
            xs = torch.cat((xs, concepts.squeeze(1)), dim=0)
            ys = torch.cat((ys, data.y.squeeze(1)), dim=0)
            train_samples += data.y.shape[0]

        for data in test_loader:
            # data.x: [batch_size, nodes_per_graph, input_feature_size/num_colors]
            # data.y: [batch_size, 1]
            concepts, _, _ = self.graph_part(data.to(self.device))
            xs = torch.cat((xs, concepts.squeeze(1)), dim=0)
            ys = torch.cat((ys, data.y.squeeze(1)), dim=0)

        train_mask = torch.arange(train_samples, device=self.device, dtype=torch.long)
        test_mask = torch.arange(ys.shape[0] - train_samples, device=self.device, dtype=torch.long) + train_samples
        concept_names = [f"c{i}" for i in range(xs.shape[1])]
        ys = torch.nn.functional.one_hot(ys)
        explanations, local_exp = entropy.explain_classes(self.dense_part, xs, ys, train_mask, test_mask,
                                                          concept_names=concept_names, class_names=class_names)
        print(explanations)
        print("----------")
        print(local_exp)
        print("##########")