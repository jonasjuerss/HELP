from __future__ import annotations

from abc import abstractmethod
from math import comb
from typing import List, Tuple, Callable

import torch
from torch.distributions import Categorical
from torch_geometric import transforms
from torch_geometric.data import Data

import data_generation.serializer as seri
from data_generation.motifs import Motif, SparseGraph


class CustomDataset(seri.ArgSerializable):

    def __init__(self, max_nodes: int, num_classes: int, num_node_features: int, class_names: List[str],
                args: dict):
        super().__init__(args)
        self.max_nodes = max_nodes
        self.num_classes = num_classes
        self.num_node_features = num_node_features
        self.dense_transform = transforms.ToDense(max_nodes)
        self.class_names = class_names

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

        if data.annotations is not None:
            size = [num_nodes - data.annotations.size(0)] + list(data.annotations.size())[1:]
            data.annotations = torch.cat([data.annotations, data.annotations.new_zeros(size)], dim=0)

        if data.pos is not None:
            size = [num_nodes - data.pos.size(0)] + list(data.pos.size())[1:]
            data.pos = torch.cat([data.pos, data.pos.new_zeros(size)], dim=0)

        return data
    def sample(self, dense=False, condition: Callable[[Data], bool]=(lambda _: True)) -> Data:
        """
        Samples a data object for a single graph and class
        :param dense: Whether to transform to dense representation
        :return:
        """
        for _ in range(100):
            sample = self._sample()
            if condition(sample):
                return self._to_dense(sample) if dense else sample
        raise TimeoutError("Could not find graph satisfying the given condition (e.g. minimum number of nodes) in 100 "
                           "attempts!")
    @abstractmethod
    def _sample(self) -> Data:
        pass


class UniqueMultipleOccurrencesMotifCategorizationDataset(CustomDataset):
    """
    Same classes as UniqueMotifCategorizationDataset but motifs may als occur multiple times.
    For categorization, it is still only relevant if a motif occurred at least once
    """

    def __init__(self, base_motif: Motif, possible_motifs: List[Motif], motif_probs: List[List[float]],
                 perturb: float = 0.00):
        """
        :param base_motif: The motif to attach others to
        :param possible_motifs: The motifs that could be present. there will be 2 ^ len(possible_motifs) classes,
        indicating for each of them whether it is present or not.
        :param motif_probs: For each motif m a list of length max_occurrences_m + 1 that indicates the probabilitity of
        it occurring 0, 1, 2, ..., max_occurrences_m times in that order
        """
        # backward compatibility:
        if len(possible_motifs) > 0 and not isinstance(possible_motifs, list):
            possible_motifs = [[1 - p, p] for p in motif_probs]

        class_names = [""]
        # Note that the first concept will have the highest class value in the end
        for m in reversed(possible_motifs):
            class_names += [prev + ("" if prev == "" else "+") + m.name for prev in class_names]
        class_names[0] = "none"

        super().__init__(base_motif.max_nodes + sum([m.max_nodes for m in possible_motifs]),
                         2 ** len(possible_motifs), base_motif.num_colors, class_names,
                         dict(base_motif=base_motif, possible_motifs=possible_motifs, motif_probs=motif_probs,
                              perturb=perturb))
        assert len(motif_probs) == len(possible_motifs)
        self.template = CustomDatasetGraphTemplate(base_motif, possible_motifs, perturb)
        self.motif_probs = motif_probs

    def _sample(self) -> Data:
        counts = []
        y = 0
        for probs in self.motif_probs:
            y *= 2
            num_motif = Categorical(probs=torch.tensor(probs)).sample()
            if num_motif > 0:
                y += 1
            counts.append(num_motif)
        graph = self.template.sample(counts)
        return Data(x=graph.x, edge_index=graph.edge_index, y=torch.tensor([y]), num_nodes=graph.num_nodes(),
                    annotations=graph.annotations)

class UniqueMotifCategorizationDataset(CustomDataset):

    def __init__(self, base_motif: Motif, possible_motifs: List[Motif], motif_probs: List[float], perturb: float = 0.0):
        """
        :param base_motif: The motif to attach others to
        :param possible_motifs: The motifs that could be present. there will be 2 ^ len(possible_motifs) classes,
        indicating for each of them whether ir is present or not.
        :param motif_probs: The probability of each motif to be present in the graph
        """
        class_names = [""]
        for m in reversed(possible_motifs):
            class_names += [prev + ("" if prev == "" else "+") + m.name for prev in class_names]
        class_names[0] = "None"

        super().__init__(base_motif.max_nodes + sum([m.max_nodes for m in possible_motifs]),
                         2 ** len(possible_motifs), base_motif.num_colors, class_names,
                         dict(base_motif=base_motif, possible_motifs=possible_motifs, motif_probs=motif_probs,
                              perturb=perturb))
        assert len(motif_probs) == len(possible_motifs)
        self.template = CustomDatasetGraphTemplate(base_motif, possible_motifs, perturb)
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
        return Data(x=graph.x, edge_index=graph.edge_index, y=torch.tensor([y]), num_nodes=graph.num_nodes(),
                    annotations=graph.annotations)


class CustomDatasetGraphTemplate(seri.ArgSerializable):
    def __init__(self, base_motif: Motif, possible_motifs: List[Motif], perturb: float):
        """
        A generator for graphs that consist of a base motif with certain other feature motifs attached to the nodes of this graph
        :param base_motif: The motif to use as a base graph
        :param possible_motifs: other motifs to attach to it
        :param perturb: There will be binom(num_nodes, perturb) random edges added (so expected value is num_nodes * perturb)
        """
        super().__init__(dict(base_motif=base_motif, possible_motifs=possible_motifs, perturb=perturb))
        self.base_motif = base_motif
        self.possible_motifs = possible_motifs
        self.perturb = perturb

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
        graph.perturb(self.perturb)
        return graph

class HierarchicalMotifGraphTemplate(seri.ArgSerializable):
    def __init__(self, highlevel_motifs: List[Motif], lowlevel_motifs: List[Motif], highlevel_motif_probs: List[float],
                 lowlevel_motif_probs: List[float], perturb: float):
        """
        A generator for graphs consisting of a high-level motif where each node itself is a low-level motif
        :param perturb: There will be binom(num_nodes, perturb) random edges added (so expected value is num_nodes * perturb)
        """
        super().__init__(dict(highlevel_motifs=highlevel_motifs, lowlevel_motifs=lowlevel_motifs,
                              highlevel_motif_probs=highlevel_motif_probs, lowlevel_motif_probs=lowlevel_motif_probs,
                              perturb=perturb))
        self.highlevel_motifs = highlevel_motifs
        self.lowlevel_motifs = lowlevel_motifs
        self.highlevel_motif_probs = highlevel_motif_probs
        self.lowlevel_motif_probs = lowlevel_motif_probs
        self.perturb = perturb
        self.num_colors = max([m.max_nodes for m in highlevel_motifs])
        if len(highlevel_motifs) != len(highlevel_motif_probs):
            raise ValueError(f"Expected number of possible high level motifs ({len(highlevel_motifs)}) to be the same as "
                             f"the number of high level probabilities ({len(highlevel_motif_probs)})")
        if len(lowlevel_motifs) != len(lowlevel_motif_probs):
            raise ValueError(f"Expected number of possible low level motifs ({len(lowlevel_motifs)}) to be the same as "
                             f"the number of low level probabilities ({len(lowlevel_motif_probs)})")

    def sample(self, highlevel_index: int, lowlevel_assignments: List[int]) -> SparseGraph:
        graph = self.highlevel_motifs[highlevel_index].sample()
        graph.x = torch.empty(graph.num_nodes(), self.num_colors)

        if graph.num_nodes() != len(lowlevel_assignments):
            raise ValueError(f"Expected sampled high level motif to have exactly as many nodes "
                             f"({graph.num_nodes()}) as low level assignments given "
                             f"({len(lowlevel_assignments)})!")

        for i, lowlevel in enumerate(lowlevel_assignments):
            low_graph = self.lowlevel_motifs[lowlevel].sample()
            low_graph.x = torch.zeros(low_graph.num_nodes(), self.num_colors)
            low_graph.x[:, i] = 1
            graph.replace_node_with_graph(i, low_graph)
        graph.perturb(self.perturb)
        return graph
class UniqueHierarchicalMotifDataset(CustomDataset):
    """
    Samples a highlevel graph consisting of one motif per low-level node. Colors of lowlevel motifs will be overwritten
    with the index of the high-level node.
    For simplicity, classes are given as class * num_lowlevel_motifs + sum(2 ** present_lowlevel_motifs - 1). Note that this
    yields some empty classes if there are highlevel motifs with less nodes than the total number of lowlevel motifs
    """
    def __init__(self, highlevel_motifs: List[Motif], lowlevel_motifs: List[Motif], highlevel_motif_probs: List[float],
                 lowlevel_motif_probs: List[float], perturb: float = 0.0):
        """

        :param highlevel_motifs:
        :param lowlevel_motifs: It is assumed that each lowlevel motif has exactly motif.max_num_nodes nodes for any sample
        :param highlevel_motif_probs:
        :param lowlevel_motif_probs:
        :param perturb:
        """

        # Note: here we use sum(2 ** index), so lower indices will yield lower numbers and we don't need to reverse
        low_class_names = [""]
        for m in lowlevel_motifs:
            low_class_names += [prev + ("" if prev == "" else "+") + m.name for prev in low_class_names]
        low_class_names = low_class_names[1:] # At least one lowlevel motif has to be present
        class_names = []
        for m in highlevel_motifs:
            class_names += [f"{m.name}:{low}" for low in low_class_names]
        max_nodes_in_highlevel = max([m.max_nodes for m in highlevel_motifs])
        self.classes_per_highlevel = 2 ** len(lowlevel_motifs) - 1  # -1 as at least one motif has to be present
        super().__init__(max_nodes_in_highlevel * max([m.max_nodes for m in lowlevel_motifs]),
                         len(highlevel_motifs) * self.classes_per_highlevel, max_nodes_in_highlevel, class_names,
                         dict(highlevel_motifs=highlevel_motifs, lowlevel_motifs=lowlevel_motifs,
                              highlevel_motif_probs=highlevel_motif_probs, lowlevel_motif_probs=lowlevel_motif_probs,
                              perturb=perturb))
        self.template = HierarchicalMotifGraphTemplate(highlevel_motifs, lowlevel_motifs, highlevel_motif_probs,
                                                       lowlevel_motif_probs, perturb)
        self.highlevel_motifs = highlevel_motifs
        self.lowlevel_motifs = lowlevel_motifs
        self.highlevel_distr = Categorical(torch.tensor(highlevel_motif_probs))
        self.lowlevel_distr = Categorical(torch.tensor(lowlevel_motif_probs))
        # classes_per_highlevel = [0] + [sum([comb(m.max_nodes, i)
        #                                     for i in range(max(m.max_nodes, len(lowlevel_motifs)) + 1)])
        #                                for m in highlevel_motifs]


    def _sample(self) -> Data:
        highlevel_index = self.highlevel_distr.sample().item()
        lowlevel_indices = self.lowlevel_distr.sample((self.highlevel_motifs[highlevel_index].max_nodes, ))
        graph = self.template.sample(highlevel_index, lowlevel_indices.tolist())
        y = self.classes_per_highlevel * highlevel_index + torch.sum(2 ** torch.unique(lowlevel_indices)) - 1
        return Data(x=graph.x, edge_index=graph.edge_index, y=torch.tensor([y]), num_nodes=graph.num_nodes(),
                    annotations=graph.annotations)