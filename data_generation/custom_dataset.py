from __future__ import annotations

from abc import abstractmethod
from math import comb
from typing import List, Tuple, Callable

import numpy as np
import torch
from torch.distributions import Categorical
from torch_geometric import transforms
from torch_geometric.data import Data

import data_generation.serializer as seri
from data_generation.motifs import Motif, SparseGraph
from graphutils import data_to_dense


class CustomDataset(seri.ArgSerializable):

    def __init__(self, max_nodes: int, num_classes: int, num_node_features: int, class_names: List[str],
                args: dict):
        super().__init__(args)
        self.max_nodes = max_nodes
        self.num_classes = num_classes
        self.num_node_features = num_node_features
        self.dense_transform = transforms.ToDense(max_nodes)
        self.class_names = class_names

    def sample(self, dense=False, condition: Callable[[Data], bool]=(lambda _: True)) -> Data:
        """
        Samples a data object for a single graph and class
        :param dense: Whether to transform to dense representation
        :return:
        """
        for _ in range(100):
            sample = self._sample()
            if condition(sample):
                return data_to_dense(sample, max_nodes=self.max_nodes) if dense else sample
        raise TimeoutError("Could not find graph satisfying the given condition (e.g. minimum number of nodes) in 100 "
                           "attempts!")
    @abstractmethod
    def _sample(self) -> Data:
        pass


class SimpleMotifCategorizationDataset(CustomDataset):

    def __init__(self, motifs: List[Motif]):
        super().__init__(max(m.max_nodes for m in motifs), len(motifs), motifs[0].num_colors, [m.name for m in motifs],
                         dict(motifs=motifs))
        self.motifs = motifs

    def _sample(self) -> Data:
        y = np.random.randint(self.num_classes)
        graph = self.motifs[y].sample()
        return Data(x=graph.x, edge_index=graph.edge_index, y=torch.tensor([y]), num_nodes=graph.num_nodes(),
                    annotations=graph.annotations)


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
                 lowlevel_motif_probs: List[float], recolor_lowlevel: bool, one_hot_color: bool,
                 num_intermediate_nodes: int, randomize_colors: bool, perturb: float):
        """
        A generator for graphs consisting of a high-level motif where each node itself is a low-level motif
        :param perturb: There will be binom(num_nodes, perturb) random edges added (so expected value is num_nodes * perturb)
        """
        super().__init__(dict(highlevel_motifs=highlevel_motifs, lowlevel_motifs=lowlevel_motifs,
                              highlevel_motif_probs=highlevel_motif_probs, lowlevel_motif_probs=lowlevel_motif_probs,
                              recolor_lowlevel=recolor_lowlevel, one_hot_color=one_hot_color,
                              num_intermediate_nodes=num_intermediate_nodes, randomize_colors=randomize_colors,
                              perturb=perturb))
        self.highlevel_motifs = highlevel_motifs
        self.lowlevel_motifs = lowlevel_motifs
        self.highlevel_motif_probs = highlevel_motif_probs
        self.lowlevel_motif_probs = lowlevel_motif_probs
        self.recolor_lowlevel = recolor_lowlevel
        self.one_hot_color = one_hot_color
        self.num_intermediate_nodes = num_intermediate_nodes
        self.randomize_colors = randomize_colors
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
        # One edge per original undirected edge pair
        edge_index_orig = graph.edge_index[:, graph.edge_index[0] >= graph.edge_index[1]]
        if self.recolor_lowlevel:
            graph.x = torch.empty(graph.num_nodes(), self.num_colors if self.one_hot_color else 1)


        if graph.num_nodes() != len(lowlevel_assignments):
            raise ValueError(f"Expected sampled high level motif to have exactly as many nodes "
                             f"({graph.num_nodes()}) as low level assignments given "
                             f"({len(lowlevel_assignments)})!")
        if self.one_hot_color:
            if self.randomize_colors:
                color_assignments = torch.randperm(len(lowlevel_assignments))
            else:
                color_assignments = torch.arange(len(lowlevel_assignments))
        else:
            # distribution uniform from {0.00, 0.01, ..., 0.99} (following: https://arxiv.org/pdf/2002.03155.pdf)
            # BUT: without replacement to make sure things are actually always distinguishable
            DISTR_SIZE = 100
            assert DISTR_SIZE >= self.num_colors + 1
            color_assignments = torch.multinomial(torch.ones(DISTR_SIZE), self.num_colors + 1) / DISTR_SIZE
        for i, lowlevel in enumerate(lowlevel_assignments):
            low_graph = self.lowlevel_motifs[lowlevel].sample()
            if self.recolor_lowlevel:
                if self.one_hot_color:
                    low_graph.x = torch.zeros(low_graph.num_nodes(), self.num_colors)
                    low_graph.x[:, color_assignments[i]] = 1
                else:
                    low_graph.x[:, 0] = color_assignments[i]
            graph.replace_node_with_graph(i, low_graph)
        if self.num_intermediate_nodes > 0:
            if self.one_hot_color:
                graph.expand_feature_dim(1)
                feature = torch.zeros(graph.num_features)
                feature[-1] = 1
            else:
                feature = torch.tensor([color_assignments[-1]])
            for i in range(edge_index_orig.shape[1]):
                last_node = edge_index_orig[1, i]
                # Obviously, repeatedly insert is inefficient but this is only done once during startup with negligible
                # time consumption anyway
                for j in range(self.num_intermediate_nodes):
                    graph.insert_node_on_edge(edge_index_orig[0, i], last_node, feature)
                    last_node = graph.num_nodes() - 1 # the node that just got inserted

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
                 lowlevel_motif_probs: List[float], recolor_lowlevel: bool = True, one_hot_color: bool = True,
                 insert_intermediate_nodes: bool = False, num_intermediate_nodes: int = 0, randomize_colors: bool = False, perturb: float = 0.0):
        """

        :param highlevel_motifs:
        :param lowlevel_motifs: It is assumed that each lowlevel motif has exactly motif.max_num_nodes nodes for any sample
        :param highlevel_motif_probs:
        :param lowlevel_motif_probs:
        :param perturb:
        """

        # Note: here we use sum(2 ** index), so lower indices will yield lower numbers and we don't need to reverse
        if insert_intermediate_nodes:
            print("Using legacy parameter insert_intermediate_nodes to set num_intermediate_nodes=1!")
            num_intermediate_nodes = 1

        low_class_names = [""]
        for m in lowlevel_motifs:
            low_class_names += [prev + ("" if prev == "" else "+") + m.name for prev in low_class_names]
        low_class_names = low_class_names[1:]  # At least one lowlevel motif has to be present
        class_names = []
        for m in highlevel_motifs:
            class_names += [f"{m.name}:{low}" for low in low_class_names]
        max_nodes_in_highlevel = max([m.max_nodes for m in highlevel_motifs])
        max_nodes_in_lowlevel = max([m.max_nodes for m in lowlevel_motifs])
        self.classes_per_highlevel = 2 ** len(lowlevel_motifs) - 1  # -1 as at least one motif has to be present
        max_nodes = max([max_nodes_in_lowlevel * m.max_nodes + m.max_edges * num_intermediate_nodes for m in highlevel_motifs])

        if one_hot_color:
            num_features = max_nodes_in_highlevel if recolor_lowlevel else lowlevel_motifs[0].num_colors
            if num_intermediate_nodes > 0:
                num_features += 1
        else:
            num_features = 1
        super().__init__(max_nodes,
                         len(highlevel_motifs) * self.classes_per_highlevel,
                         num_features,
                         class_names,
                         dict(highlevel_motifs=highlevel_motifs, lowlevel_motifs=lowlevel_motifs,
                              highlevel_motif_probs=highlevel_motif_probs, lowlevel_motif_probs=lowlevel_motif_probs,
                              recolor_lowlevel=recolor_lowlevel, one_hot_color=one_hot_color,
                              num_intermediate_nodes=num_intermediate_nodes, randomize_colors=randomize_colors,
                              perturb=perturb))
        self.template = HierarchicalMotifGraphTemplate(highlevel_motifs, lowlevel_motifs, highlevel_motif_probs,
                                                       lowlevel_motif_probs, recolor_lowlevel, one_hot_color,
                                                       num_intermediate_nodes, randomize_colors, perturb)
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