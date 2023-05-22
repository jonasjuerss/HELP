"""
The custom motif dataset is copied from my MPhil Thesis
"""
from __future__ import annotations

import json
from typing import Union, Tuple, List, Optional

import numpy as np
import torch
from torch import Tensor, LongTensor
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from abc import ABC, abstractmethod

import data_generation.serializer as cs
from graphutils import adj_to_edge_index
import networkx as nx
import matplotlib.pyplot as plt
import torch_geometric

class SparseGraph:
    def __init__(self, x: Tensor, edge_index: Tensor, annotations: Optional[Tensor] = None):
        self.x = x
        self.edge_index = edge_index
        self.annotations = annotations

    def merged_with(self, other: SparseGraph, merge_nodes: Union[Tuple[int, int], None] = None) -> SparseGraph:
        annotations = None
        if merge_nodes is None:
            x = torch.cat((self.x, other.x), dim=0)
            if self.annotations is not None:
                annotations = torch.cat((self.annotations, other.annotations), dim=0)
            edge_index = torch.cat((self.edge_index, other.edge_index + self.num_nodes()), dim=1)
        else:
            # leave out the merged node from new
            x = torch.cat((self.x, other.x[:merge_nodes[1], :], other.x[merge_nodes[1] + 1:, :]), dim=0)
            if self.annotations is not None:
                annotations = torch.cat((self.annotations, other.annotations[:merge_nodes[1], :],
                                         other.annotations[merge_nodes[1] + 1:, :]), dim=0)
            # adjust edge_index of new graph accordingly. Will temporarily yield negative node indices because the node will be in x_base
            edge_index = other.edge_index.copy()
            edge_index[edge_index == merge_nodes[1]] = merge_nodes[0] - self.num_nodes()
            edge_index[edge_index > merge_nodes[1]] -= 1  # adjust all node indices after the removed node
            edge_index = torch.cat((self.edge_index, edge_index + self.num_nodes()), dim=1)
        return SparseGraph(x, edge_index, annotations)

    def add_nodes(self, features: Tensor, annotations: Optional[Tensor] = None) -> None:
        if len(features.shape) == 1:
            features = features[None, :]
        elif len(features.shape) != 2:
            raise ValueError(f"Expected feature vector to be of dimension 1 or 2 but got {len(features.shape)}!")

        if features.shape[1] == self.x.shape[1]:
            self.x = torch.cat((self.x, features), dim=0)
            if self.annotations is not None:
                self.annotations = torch.cat((self.annotations, annotations), dim=1)
        else:
            raise ValueError(f"Expected {self.x.shape[1]} features but got {features.shape[0]}!")

    def add_edges(self, edges: List[List[int]]) -> None:
        """
        :param edges: Note: this is a list of edges (e.g. [[0, 1], [1, 0]]) would add a bidirectional edge between nodes
        0 and 1. It is not a slice of an edge_index tensor
        """
        self.edge_index = torch.cat((self.edge_index, torch.tensor(edges, dtype=torch.long).T), dim=1)

    def add_edges_if_not_exist_list(self, edges: List[List[int]]) -> None:
        """
        :param edges: Note: this is a list of edges (e.g. [[0, 1], [1, 0]]) would add a bidirectional edge between nodes
        0 and 1. It is not a slice of an edge_index tensor
        """
        # [2, num_edges]
        added_edge_index = torch.tensor(edges, dtype=torch.long).T
        self.add_edges_if_not_exist_edge_index(added_edge_index)

    def add_edges_if_not_exist_edge_index(self, edge_index: Tensor) -> None:
        for i in range(edge_index.shape[1]):
            if not torch.any(torch.all(self.edge_index == edge_index[:, i:i+1], dim=0)):
                self.edge_index = torch.cat((self.edge_index, edge_index[:, i:i+1]), dim=1)

    def replace_node_with_graph(self, node_index: int, graph: SparseGraph) -> None:
        """
        Replaces the node at the given index with the given graph. In particular the features of the given node
        are overwritten with the features of the first node in the given graph and all other nodes are appended along
        with an appropriate edge_index

        :param node_index: Index of the node to replace with the graph
        :param graph: graph to replace the node with
        """
        if graph.num_features != self.num_features:
            raise ValueError(f"Expected same number of features for current graph ({self.num_features}) and attached "
                             f"graph ({graph.num_features})!")

        num_edges_prev = self.num_edges
        self.edge_index = torch.cat((self.edge_index, graph.edge_index), dim=1)
        self.edge_index[:, num_edges_prev:][self.edge_index[:, num_edges_prev:] != 0] += self.num_nodes() - 1
        self.edge_index[:, num_edges_prev:][self.edge_index[:, num_edges_prev:] == 0] = node_index

        self.x[node_index, :] = graph.x[0, :]
        self.x = torch.cat((self.x, graph.x[1:, :]), dim=0)
        if self.annotations is not None:
            self.annotations[node_index] = graph.annotations[0]
            self.annotations = torch.cat((self.annotations, graph.annotations[1:]), dim=0)


    def remove_edge(self, from_node: int, to_node: int) -> bool:
        mask = torch.any(self.edge_index != torch.tensor([[from_node], [to_node]]), dim=0)
        self.edge_index = self.edge_index[:, mask]
        return not torch.all(mask)

    def insert_node_on_edge(self, from_node: int, to_node: int, features: torch.Tensor, directed=False,
                            annotation: Optional[Tensor] = None):
        if not self.remove_edge(from_node, to_node) or (not directed and not self.remove_edge(to_node, from_node)):
            raise ValueError(f"No {'' if directed else 'un'}directed edge between nodes {from_node} and {to_node} "
                             f"found!")
        added_edges = torch.tensor([[from_node, self.num_nodes()], [self.num_nodes(), to_node]]).T
        if not directed:
            added_edges = torch.cat((added_edges, torch.flip(added_edges, (0, ))), dim=1)
        self.edge_index = torch.cat((self.edge_index, added_edges), dim=1)
        self.add_nodes(features, annotation)

    def expand_feature_dim(self, num_dims: int):
        self.x = torch.cat((self.x, torch.zeros(self.x.shape[0], num_dims)), dim=1)

    def perturb(self, prob: float) -> None:
        """
        Adds roughly num_edges ~ Bernoulli(num_nodes, prob) random undirected edges to the graph
        """
        if prob == 0:
            return
        num_added_edges = int(torch.distributions.binomial.Binomial(self.num_nodes(),
                                                                    torch.tensor(prob)).sample().item())
        edge_index = torch.randint(0, self.num_nodes(), (2, num_added_edges))
        edge_index = torch.cat((edge_index, edge_index[[1, 0], :]), dim=1)
        # Edge index may contain duplicates but we deal with this in
        self.add_edges_if_not_exist_edge_index(edge_index)
    def num_nodes(self):
        return self.x.shape[0]

    @property
    def num_features(self):
        return self.x.shape[1]

    @property
    def num_edges(self):
        return self.edge_index.shape[1]

    def render(self, filename=None):
        data = Data(x=self.x, edge_index=self.edge_index)
        g = torch_geometric.utils.to_networkx(data, to_undirected=False)
        color = torch.mean(self.x * torch.arange(self.num_features)[None, :], dim=-1)
        nx.draw(g, node_color=color, with_labels=True)
        if filename is not None:
            plt.savefig(filename)
        plt.show()


class Motif(cs.ArgSerializable):
    def __init__(self, num_colors: int, max_nodes: int, max_edges: int, args: dict):
        """
        :param num_colors: The total number of colors in the graph
        :param max_nodes: the maximum number of nodes a sample of this motif can have
        :param args: All keyword arguments the constructor of the subclass took. This is used for serialization
        """
        super().__init__(args)
        self.num_colors = num_colors
        self.max_nodes = max_nodes
        self.max_edges = max_edges

    @abstractmethod
    def sample(self) -> SparseGraph:
        pass

    @property
    def name(self):
        return self.__class__.__name__[:-5]


class HouseMotif(Motif):

    def __init__(self, roof_colors: List[int], basement_colors: List[int], num_colors: int,
                 roof_annotation: Optional[int] = None, basement_annotation: Optional[int] = None):
        """
          0
         / \
        1---2
        |   |
        3---4
        :param roof_colors: Possible colors for the three roof nodes (will be chosen uniformly at random on sample())
        :param basement_colors: Possible colors for the two basement nodes (will be chosen uniformly at random on sample())
        :param num_colors: number of different colors in the overall graph
        """
        super().__init__(num_colors, 5, 2 * 6,
                         dict(roof_colors=roof_colors, basement_colors=basement_colors, num_colors=num_colors,
                              roof_annotation=roof_annotation, basement_annotation=basement_annotation))
        self.roof_colors = roof_colors
        self.basement_colors = basement_colors
        self.roof_annotation = roof_annotation
        self.basement_annotation = basement_annotation

    def sample(self) -> SparseGraph:
        roof_color = _random_list_entry(self.roof_colors)
        basement_color = _random_list_entry(self.basement_colors)
        x = torch.zeros((5, self.num_colors))
        x[:3, roof_color] = 1
        x[3:, basement_color] = 1
        if self.roof_annotation and self.basement_annotation:
            annotations = torch.empty((5,), dtype=torch.long)
            annotations[:3] = self.roof_annotation
            annotations[3:] = self.basement_annotation
        else:
            annotations = None
        edge_index = torch.tensor([[0, 1], [0, 2], [1, 2], [1, 3], [2, 4], [3, 4]], dtype=torch.long).T
        return SparseGraph(x, to_undirected(edge_index), annotations)

class SplitHexagon(Motif):

    def __init__(self, colors: List[int], num_colors: int, annotation: Optional[int] = None):
        """
          0
         / \
        1---2
        |   |
        3---4
         \ /
          5
        """
        super().__init__(num_colors, 6, 2 * 8,
                         dict(colors=colors, num_colors=num_colors, annotation=annotation))
        self.colors = torch.tensor(colors)
        self.annotation = annotation

    def sample(self) -> SparseGraph:
        color = self.colors[torch.randperm(self.colors.shape[0])[:6]]
        x = torch.zeros((6, self.num_colors))
        x[torch.arange(6), color] = 1
        if self.annotation:
            annotations = torch.full((5,), self.annotation, dtype=torch.long)
        else:
            annotations = None
        edge_index = torch.tensor([[0, 1], [0, 2], [1, 2], [1, 3], [2, 4], [3, 4], [3, 5], [4, 5]], dtype=torch.long).T
        return SparseGraph(x, to_undirected(edge_index), annotations)


class CrossHexagon(Motif):

    def __init__(self, colors: List[int], num_colors: int, annotation: Optional[int] = None):
        """
          0
         / \
        1   2
        | X |
        3   4
         \ /
          5
        """
        super().__init__(num_colors, 6, 2 * 8,
                         dict(colors=colors, num_colors=num_colors, annotation=annotation))
        self.colors = torch.tensor(colors)
        self.annotation = annotation

    def sample(self) -> SparseGraph:
        color = self.colors[torch.randperm(self.colors.shape[0])[:6]]
        x = torch.zeros((6, self.num_colors))
        x[torch.arange(6), color] = 1
        if self.annotation:
            annotations = torch.full((5,), self.annotation, dtype=torch.long)
        else:
            annotations = None
        edge_index = torch.tensor([[0, 1], [0, 2], [1, 4], [1, 3], [2, 4], [3, 2], [3, 5], [4, 5]], dtype=torch.long).T
        return SparseGraph(x, to_undirected(edge_index), annotations)


class IntermediateNodeMotif(Motif):
    def __init__(self, motif: Motif, num_intermediates: int, color: int):
        super().__init__(motif.num_colors, motif.max_nodes + motif.max_edges * num_intermediates,
                         motif.max_edges * (num_intermediates + 1),
                         dict(motif=motif, num_intermediates=num_intermediates, color=color))
        self.motif = motif
        self.num_intermediates = num_intermediates
        self.color = color

    def sample(self) -> SparseGraph:
        graph = self.motif.sample()
        edge_index_orig = graph.edge_index[:, graph.edge_index[0] >= graph.edge_index[1]]
        feature = torch.zeros(self.num_colors)
        feature[self.color] = 1
        for i in range(edge_index_orig.shape[1]):
            last_node = edge_index_orig[1, i]
            # Obviously, repeatedly insert is inefficient but this is only done once during startup with negligible
            # time consumption anyway
            for j in range(self.num_intermediates):
                graph.insert_node_on_edge(edge_index_orig[0, i], last_node, feature)
                last_node = graph.num_nodes() - 1  # the node that just got inserted
        return graph



class FullyConnectedMotif(Motif):
    def __init__(self, num_nodes: int, colors: List[int], num_colors: int, annotation: Optional[int] = None):
        super().__init__(num_colors, num_nodes, 2 * num_nodes * (num_nodes - 1) , dict(num_nodes=num_nodes, colors=colors, num_colors=num_colors,
                                                     annotation=annotation))
        self.colors = colors
        self.num_nodes = num_nodes
        self.annotation = annotation

    def sample(self) -> SparseGraph:
        color = _random_list_entry(self.colors)
        x = torch.zeros((self.num_nodes, self.num_colors))
        x[:, color] = 1
        adj = torch.ones((self.num_nodes, self.num_nodes), dtype=torch.long)
        node_indices = torch.arange(self.num_nodes)
        adj[node_indices, node_indices] = 0
        edge_index, _, _ = adj_to_edge_index(adj)
        annotations = None if self.annotation is None else torch.ones(x.shape[0], dtype=torch.long) * self.annotation
        return SparseGraph(x, edge_index, annotations)

    @property
    def name(self):
        if self.num_nodes == 3:
            return "Triangle"
        elif self.num_nodes == 4:
            return "FCSquare"
        elif self.num_nodes == 5:
            return "FCPentagon"
        elif self.num_nodes == 6:
            return "FCHexagon"
        else:
            return f"FullyConnected ({self.num_nodes})"


class CircleMotif(Motif):
    def __init__(self, num_nodes: int, colors: List[int], num_colors: int, nodes_upper_bound: Optional[int] = None,
                 num_nodes_step: int = 1, annotation: Optional[int] = None):
        super().__init__(num_colors, num_nodes if nodes_upper_bound is None else nodes_upper_bound,
                         2 * (num_nodes if nodes_upper_bound is None else nodes_upper_bound),
                         dict(num_nodes=num_nodes, colors=colors, num_colors=num_colors,
                              nodes_upper_bound=nodes_upper_bound, num_nodes_step=num_nodes_step, annotation=annotation))
        self.colors = colors
        self.num_nodes = num_nodes
        self.annotation = annotation
        self.nodes_upper_bound = nodes_upper_bound
        self.num_nodes_step = num_nodes_step

    def sample(self) -> SparseGraph:
        num_nodes = self.num_nodes if self.nodes_upper_bound is None\
            else self.num_nodes + self.num_nodes_step * np.random.randint(1 + ((self.nodes_upper_bound - self.num_nodes)
                                                                               // self.num_nodes_step))
        color = _random_list_entry(self.colors)
        x = torch.zeros((num_nodes, self.num_colors))
        x[:, color] = 1
        node_range = torch.arange(num_nodes)
        edge_index = torch.stack((node_range, torch.remainder(node_range + 1, num_nodes)), dim=0)
        edge_index = torch.cat((edge_index, torch.flip(edge_index, dims=(0, ))), dim=1)
        annotations = None if self.annotation is None else torch.ones(x.shape[0], dtype=torch.long) * self.annotation
        return SparseGraph(x, edge_index, annotations)

    @property
    def name(self):
        return f"Circle ({self.num_nodes}" +\
            ("" if self.max_nodes is None else f"-{self.max_nodes}, step={self.num_nodes_step}")


class SetMotif(Motif):

    def __init__(self, elements: List[Motif]):
        super().__init__(elements[0].num_colors, sum(e.max_nodes for e in elements),
                         sum(e.max_edges for e in elements),
                         dict(elements=elements))
        self.elements = elements

    def sample(self) -> SparseGraph:
        res = self.elements[0].sample()
        for e in self.elements[1:]:
            res = res.merged_with(e.sample())
        return res

    @property
    def name(self):
        return f"Set ({', '.join(e.name for e in self.elements)})"


class ReplicationMotif(Motif):
    def __init__(self, motif: Motif, num_replications: int):
        super().__init__(motif.num_colors, num_replications * motif.max_nodes,
                         num_replications * motif.max_edges,
                         dict(motif=motif, num_replications=num_replications))
        self.motif = motif
        self.num_replications = num_replications

    def sample(self) -> SparseGraph:
        first = self.motif.sample()
        res = first
        for i in range(self.num_replications - 1):
            res = res.merged_with(first)
        return res

    @property
    def name(self):
        return f"({self.num_replications} x {self.motif.name})"

class BinaryTreeMotif(Motif):
    def __init__(self, max_depth: int, colors: List[int], num_colors: int, random: bool = True,
                 annotation: Optional[int] = None):
        super().__init__(num_colors, (2 ** (max_depth + 1)) - 1, 2 * ((2 ** (max_depth + 1)) - 2),
                         dict(max_depth=max_depth, colors=colors, num_colors=num_colors, random=random))
        self.max_depth = max_depth
        self.colors = colors
        self.random = random
        self.annotation = annotation

    def sample(self):
        """
        Idea: each path from root to leaves has length 1,...,max_depth with same probability
        """
        color = _random_list_entry(self.colors)
        tree = self._random_binary_tree(self.max_depth, color)
        if self.annotation is not None:
            tree.annotations = torch.ones(tree.num_nodes(), dtype=torch.long) * self.annotation
        return tree

    def _random_binary_tree(self, max_depth: int, color: int) -> SparseGraph:
        x = torch.zeros((1, self.num_colors))
        x[:, color] = 1
        edge_index = torch.tensor([[], []], dtype=torch.long)
        result = SparseGraph(x, edge_index)
        if self.max_depth == 0:
            return result
        if not self.random or torch.rand(()) < (max_depth - 1) / max_depth:
            left = self._random_binary_tree(max_depth - 1, color)
            result = result.merged_with(left)
            result.add_edges([[0, 1], [1, 0]])
        if not self.random or torch.rand(()) < (max_depth - 1) / max_depth:
            num_nodes = result.num_nodes()
            right = self._random_binary_tree(max_depth - 1, color)
            result = result.merged_with(right)
            result.add_edges([[0, num_nodes], [num_nodes, 0]])
        return result


def _random_list_entry(list: List[int]):
    return list[torch.randint(len(list), (1,))]