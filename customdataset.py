from abc import ABC, abstractmethod

import torch
from typing import Tuple, Union, List
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected  # just adds the other directions
import torch_geometric
import networkx as nx
from torch_geometric import transforms

from motifs import Motif, SparseGraph


class CustomDataset(ABC):

    def __init__(self, max_nodes: int, num_classes: int):
        self.max_nodes = max_nodes
        self.num_classes = num_classes

    @abstractmethod
    def sample(self) -> Data:
        pass


class UniqueMotifCategorizationDataset(CustomDataset):

    def __init__(self, base_motif: Motif, possible_motifs: List[Motif], motif_probs: List[float]):
        super().__init__(base_motif.max_nodes + sum([m.max_nodes for m in possible_motifs]), 2 ** len(possible_motifs))
        assert len(motif_probs) == len(possible_motifs)
        self.template = CustomDatasetGraphTemplate(base_motif, possible_motifs)
        self.motif_probs = motif_probs

    def sample(self) -> Data:
        counts = []
        y = 0
        for prob in self.motif_probs:
            y *= 2
            if torch.randint(2, (1,)) < prob:
                y += 1
                counts.append(1)
            else:
                counts.append(0)
        graph = self.template.sample(counts)
        return Data(x=graph.x, edge_index=graph.edge_index, y=torch.tensor(y))


class CustomDatasetGraphTemplate:
    def __init__(self, base_motif: Motif, possible_motifs: List[Motif]):
        """
        A generator for graphs that consist of a base motif with certain other feature motifs attached to the nodes of this graph
        :param base_motif: The motif to use as a base graph
        :param possible_motifs: other motifs to attach to it
        """
        super().__init__(base_motif.num_colors)  # should be the same for all motifs
        self.base_motif = base_motif
        self.possible_motifs = possible_motifs

    def sample(self, motif_counts: List[int]) -> SparseGraph:
        if len(motif_counts) != len(self.possible_motifs):
            raise ValueError(
                f"Expected count for each of the {len(self.possible_motifs)} motifs but got {len(motif_counts)} counts!")

        graph = self.base_motif.sample()
        num_nodes_orig = graph.num_nodes()
        for num, motif in zip(motif_counts, self.possible_motifs):
            for i in range(num):
                connect_loc = torch.randint(num_nodes_orig, ())
                num_nodes_prev = graph.num_nodes()
                graph = graph.merged_with(motif.sample())
                graph.add_edges([[connect_loc, num_nodes_prev], [num_nodes_prev, connect_loc]])
        return graph

    def from_dict(self, params):
        pass
