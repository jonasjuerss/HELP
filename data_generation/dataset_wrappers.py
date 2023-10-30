import abc
import random
import traceback
from typing import Optional, List, Type

import torch
from torch_geometric.data import Dataset
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import ToDense, Compose, Constant

import data_generation.serializer as seri
from color_utils import ColorUtils
from data_generation.custom_dataset import CustomDataset, SimpleMotifCategorizationDataset
from data_generation.motifs import ReplicationMotif, CircleMotif
from data_generation.transforms import RemoveEdgeFeatures
from graphutils import data_to_dense


class DatasetWrapper(seri.ArgSerializable, abc.ABC):
    def __init__(self, num_classes: int, num_node_features: int, is_directed: bool, max_nodes_per_graph: int,
                 args: dict,
                 class_names: Optional[List[str]] = None):
        super().__init__(args)
        ColorUtils.reset()
        self._dataset = None
        self.num_classes = num_classes
        self.num_node_features = num_node_features
        self.is_directed = is_directed
        self.max_nodes_per_graph = max_nodes_per_graph
        self._num_nodes_total = None
        self._num_edges_total = None
        if class_names is None:
            self.class_names = [f"Class {i}" for i in range(num_classes)]
        else:
            if len(class_names) != num_classes:
                raise ValueError(f"Got {len(class_names)} class names for {num_classes} classes!")
            self.class_names = class_names

    @abc.abstractmethod
    def _get_dataset(self, dense: bool, min_nodes: int):
        pass

    def get_dataset(self, dense: bool, min_nodes: int):
        if self._dataset is None:
            self._dataset = self._get_dataset(dense, min_nodes)
        return self._dataset

    @property
    def num_nodes_total(self) -> int:
        """
        Important: note that this will shuffle the dataset
        """
        if self._dataset is None:
            raise RuntimeError("Number of nodes cannot be calculated as dataset hasn't been generated yet.")
        if self._num_nodes_total is None:
            self._num_nodes_total = sum([d.num_nodes for d in self._dataset])
        return self._num_nodes_total

    @property
    def num_edges_total(self) -> int:
        """
        Important: note that this will shuffle the dataset
        """
        if self._dataset is None:
            raise RuntimeError("Number of edges cannot be calculated as dataset hasn't been generated yet.")
        if self._num_edges_total is None:
            try:
                self._num_edges_total = sum([d.edge_index.shape[0] for d in self._dataset])
            except:
                print("Could not get number of edges. Note that this is only possible for sparse data.")
                traceback.print_exc()
        return self._num_edges_total


class CustomDatasetWrapper(DatasetWrapper, abc.ABC):

    def __init__(self, sampler: CustomDataset, num_samples=4 * (512 + 128)):
        super().__init__(sampler.num_classes, sampler.num_node_features, False, sampler.max_nodes,
                         dict(sampler=sampler, num_samples=num_samples),
                         sampler.class_names)
        self.sampler = sampler
        self.num_samples = num_samples
        ColorUtils.set_feature_colors(torch.Tensor([
            [18, 18, 18],
            [52, 152, 219]]).float())

    def _get_dataset(self, dense: bool, min_nodes: int):
        def condition(d):
            return d.num_nodes >= min_nodes

        return [self.sampler.sample(dense, condition) for _ in range(self.num_samples)]


# For convenience
class ExpressivityWrapper(CustomDatasetWrapper):
    def __init__(self):
        super().__init__(SimpleMotifCategorizationDataset(
            [ReplicationMotif(CircleMotif(3, [0], 1, 20, 1), 2),
             CircleMotif(6, [0], 1, 40, 2)]))


class PyGWrapper(DatasetWrapper, abc.ABC):

    def __init__(self, dataset_class: Type[Dataset], remove_edge_fts: bool, is_directed: bool, dataset_kwargs: dict,
                 args: dict, class_names: Optional[List[str]] = None):
        self.dummy_dataset = dataset_class(**dataset_kwargs)
        super().__init__(self.dummy_dataset.num_classes, self.dummy_dataset.num_node_features, is_directed,
                         max([d.num_nodes for d in self.dummy_dataset]), args,
                         class_names)
        self.dataset_class = dataset_class
        self.dataset_kwargs = dataset_kwargs
        self.remove_edge_fts = remove_edge_fts

    def _get_dataset(self, dense: bool, min_nodes: int):
        kwargs = dict()
        transforms = []
        if min_nodes > 0:
            def condition(d):
                return d.num_nodes >= min_nodes

            kwargs["pre_filter"] = condition

        for d in self.dummy_dataset:
            break
        if not hasattr(d, "x") or d.x is None:
            transforms.append(Constant())

        if self.remove_edge_fts:
            transforms.append(RemoveEdgeFeatures())

        if dense:
            transforms.append(ToDense(num_nodes=self.max_nodes_per_graph))

        if len(transforms) > 0:
            kwargs["transform"] = Compose(transforms) if len(transforms) > 1 else transforms[0]
        del self.dummy_dataset

        dataset = self.dataset_class(**self.dataset_kwargs, **kwargs)
        dataset = dataset.shuffle()
        return dataset


class PtFileWrapper(DatasetWrapper):

    def __init__(self, path: str, args: dict):
        self.data_dict = torch.load(path)
        super().__init__(self.data_dict["num_classes"], self.data_dict["num_node_features"],
                         self.data_dict.get("is_directed", False), max([d.num_nodes for d in self.data_dict["data"]]),
                         args, self.data_dict.get("class_names", None))

    def _get_dataset(self, dense: bool, min_nodes: int):
        def condition(d):
            return d.num_nodes >= min_nodes

        transform = (lambda d: data_to_dense(d, max_nodes=self.max_nodes_per_graph)) if dense else (lambda x: x)
        dataset = [transform(d) for d in self.data_dict["data"] if condition(d)]
        random.Random(torch.initial_seed()).shuffle(dataset)
        return dataset


class BBBPWrapper(PtFileWrapper):
    def __init__(self):
        super().__init__("data/bbbp.pt", dict())


class BBBPAtomWrapper(PtFileWrapper):
    def __init__(self):
        super().__init__("data/bbbp_atom.pt", dict())
        # https://github.com/DeepGraphLearning/torchdrug/issues/197
        # atom_types = [1, 5, 6, 7, 8, 9, 11, 15, 16, 17, 20, 35, 53]
        ColorUtils.feature_labels = ['H', 'B', 'C', 'N', 'O', 'F', 'Na', 'P', 'S', 'Cl', 'Ca', 'Br', 'I']
        ColorUtils.set_feature_colors(torch.Tensor([
            [52, 152, 219],
            [142, 68, 173],
            [44, 62, 80],
            [205, 220, 57],
            [231, 76, 60],
            [243, 156, 18],
            [232, 67, 147],
            [96, 125, 139],
            [142, 68, 173],
            [39, 174, 96],
            [0, 150, 136],
            [121, 85, 72],
            [127, 140, 141],
        ]).float())


class TUDatasetWrapper(PyGWrapper):
    def __init__(self, dataset_name: str, is_directed: bool, remove_edge_fts: bool = False, args=None,
                 class_names: Optional[List[str]] = None):
        # Note that the downloaded files (in C:\tmp in my case on windows because I set root to /tmp/) have a README
        # explaining node and class labels
        if args is None:
            args = dict(dataset_name=dataset_name)
        super().__init__(TUDataset, remove_edge_fts, is_directed, dict(root='/tmp', name=dataset_name), args,
                         class_names)


class MutagWrapper(TUDatasetWrapper):
    def __init__(self, remove_edge_fts: bool = True):
        super().__init__("MUTAG", False, remove_edge_fts, dict(remove_edge_fts=remove_edge_fts),
                         ["not mutagenic", "mutagenic"])
        # ColorUtils.rgb_feature_colors = None
        ColorUtils.feature_labels = ['C', 'O', 'Cl', 'H', 'N', 'F', 'Br', 'S', 'P', 'I', 'Na', 'K', 'Li', 'Ca']
        ColorUtils.set_feature_colors(torch.Tensor([
            [44, 62, 80],
            [231, 76, 60],
            [39, 174, 96],
            [52, 152, 219],
            [205, 220, 57],
            [243, 156, 18],
            [121, 85, 72],
            [142, 68, 173],
            [63, 81, 181],
            [127, 140, 141],
            [232, 67, 147],
            [96, 125, 139],
            [142, 68, 173],
            [0, 150, 136]]).float())
        #         self.color_map = np.array(
        #             ['#2c3e50', '#e74c3c', '#27ae60', '#3498db', '#CDDC39', '#f39c12', '#795548', '#8e44ad', '#3F51B5',
        #              '#7f8c8d', '#e84393', '#607D8B', '#8e44ad', '#009688'])


class MutagenicityWrapper(TUDatasetWrapper):
    def __init__(self, remove_edge_fts: bool = True):
        super().__init__("Mutagenicity", False, remove_edge_fts, dict(remove_edge_fts=remove_edge_fts),
                         ["not mutagenic", "mutagenic"])
        ColorUtils.set_feature_colors(torch.Tensor([
            [44, 62, 80],
            [231, 76, 60],
            [39, 174, 96],
            [52, 152, 219],
            [205, 220, 57],
            [243, 156, 18],
            [121, 85, 72],
            [142, 68, 173],
            [63, 81, 181],
            [127, 140, 141],
            [232, 67, 147],
            [96, 125, 139],
            [142, 68, 173],
            [0, 150, 136]]).float())
        ColorUtils.feature_labels = ['C', 'O', 'Cl', 'H', 'N', 'F', 'Br', 'S', 'P', 'I', 'Na', 'K', 'Li', 'Ca']


class RedditBinaryWrapper(TUDatasetWrapper):
    def __init__(self):
        super().__init__("REDDIT-BINARY", is_directed=False, args=dict())


class EnzymesWrapper(TUDatasetWrapper):
    def __init__(self):
        super().__init__("ENZYMES", is_directed=False, args=dict())


class ProteinsWrapper(TUDatasetWrapper):
    def __init__(self):
        super().__init__("PROTEINS", is_directed=False, args=dict())
