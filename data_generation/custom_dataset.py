from __future__ import annotations

from abc import abstractmethod
from typing import List, Tuple

import torch
from torch_geometric import transforms
from torch_geometric.data import Data

import data_generation.serializer as seri
from data_generation.motifs import Motif, SparseGraph


class CustomDataset(seri.ArgSerializable):

    def __init__(self, max_nodes: int, num_classes: int, num_node_features: int, args: dict):
        super().__init__(args)
        self.max_nodes = max_nodes
        self.num_classes = num_classes
        self.num_node_features = num_node_features
        self.dense_transform = transforms.ToDense(max_nodes)

    def _to_dense(self, data: Data):
        """
        This is basically the implementation from :class:`torch_geometric.transforms.ToDense` but without
        converting the label to shape max_nodes if the graph consists of a single node
        """
        assert data.edge_index is not None

        orig_num_nodes = data.num_nodes
        if self.max_nodes is None:
            num_nodes = orig_num_nodes
        else:
            assert orig_num_nodes <= self.max_nodes
            num_nodes = self.max_nodes

        if data.edge_attr is None:
            edge_attr = torch.ones(data.edge_index.size(1), dtype=torch.float)
        else:
            edge_attr = data.edge_attr

        size = torch.Size([num_nodes, num_nodes] + list(edge_attr.size())[1:])
        adj = torch.sparse_coo_tensor(data.edge_index, edge_attr, size)
        data.adj = adj.to_dense()
        data.edge_index = None
        data.edge_attr = None

        data.mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.mask[:orig_num_nodes] = 1

        if data.x is not None:
            size = [num_nodes - data.x.size(0)] + list(data.x.size())[1:]
            data.x = torch.cat([data.x, data.x.new_zeros(size)], dim=0)

        if data.pos is not None:
            size = [num_nodes - data.pos.size(0)] + list(data.pos.size())[1:]
            data.pos = torch.cat([data.pos, data.pos.new_zeros(size)], dim=0)

        return data
    def sample(self, dense=False) -> Data:
        """
        Samples a data object for a single graph and class
        :param dense: Whether to transform to dense representation
        :return:
        """
        return self._to_dense(self._sample()) if dense else self._sample()
    @abstractmethod
    def _sample(self) -> Data:
        pass


class UniqueMotifCategorizationDataset(CustomDataset):

    def __init__(self, base_motif: Motif, possible_motifs: List[Motif], motif_probs: List[float]):
        """
        :param base_motif: The motif to attach others to
        :param possible_motifs: The motifs that could be present. there will be 2 ^ len(possible_motifs) classes,
        indicating for each of them whether ir is present or not.
        :param motif_probs: The probability of each motif to be present in the graph
        """
        super().__init__(base_motif.max_nodes + sum([m.max_nodes for m in possible_motifs]),
                         2 ** len(possible_motifs), base_motif.num_colors,
                         dict(base_motif=base_motif, possible_motifs=possible_motifs, motif_probs=motif_probs))
        assert len(motif_probs) == len(possible_motifs)
        self.template = CustomDatasetGraphTemplate(base_motif, possible_motifs)
        self.motif_probs = motif_probs

    def _sample(self) -> Data:
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
        return Data(x=graph.x, edge_index=graph.edge_index, y=torch.tensor(y)[None])


class CustomDatasetGraphTemplate(seri.ArgSerializable):
    def __init__(self, base_motif: Motif, possible_motifs: List[Motif]):
        """
        A generator for graphs that consist of a base motif with certain other feature motifs attached to the nodes of this graph
        :param base_motif: The motif to use as a base graph
        :param possible_motifs: other motifs to attach to it
        """
        super().__init__(dict(base_motif=base_motif, possible_motifs=possible_motifs))
        self.base_motif = base_motif
        self.possible_motifs = possible_motifs

    def sample(self, motif_counts: List[int]) -> SparseGraph:
        if len(motif_counts) != len(self.possible_motifs):
            raise ValueError(f"Expected count for each of the {len(self.possible_motifs)} motifs "
                             f"but got {len(motif_counts)} counts!")

        graph = self.base_motif.sample()
        num_nodes_orig = graph.num_nodes()
        for num, motif in zip(motif_counts, self.possible_motifs):
            for i in range(num):
                connect_loc = torch.randint(num_nodes_orig, ())
                num_nodes_prev = graph.num_nodes()
                graph = graph.merged_with(motif.sample())
                graph.add_edges([[connect_loc, num_nodes_prev], [num_nodes_prev, connect_loc]])
        return graph