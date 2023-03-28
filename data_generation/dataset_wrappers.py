import abc
from typing import Optional, List, Type

from torch_geometric.data import Dataset
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import ToDense, Compose, Constant

import data_generation.serializer as seri
from data_generation.custom_dataset import CustomDataset


class DatasetWrapper(seri.ArgSerializable, abc.ABC):
    def __init__(self, num_classes: int, num_node_features: int, args: dict,
                 class_names: Optional[List[str]] = None):
        super().__init__(args)
        self._dataset = None
        self.num_classes = num_classes
        self.num_node_features = num_node_features
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


class CustomDatasetWrapper(DatasetWrapper, abc.ABC):

    def __init__(self, sampler: CustomDataset, num_samples=512 + 128):
        super().__init__(sampler.num_classes, sampler.num_node_features,
                         dict(sampler=sampler, num_samples=num_samples),
                         sampler.class_names)
        self.sampler = sampler
        self.num_samples = num_samples

    def _get_dataset(self, dense: bool, min_nodes: int):
        def condition(d):
            return d.num_nodes >= min_nodes

        return [self.sampler.sample(dense, condition) for _ in range(self.num_samples)]


class PyGWrapper(DatasetWrapper, abc.ABC):

    def __init__(self, dataset_class: Type[Dataset], dataset_kwargs: dict, args: dict,
                 class_names: Optional[List[str]] = None):
        self.dummy_dataset = dataset_class(**dataset_kwargs)
        super().__init__(self.dummy_dataset.num_classes, self.dummy_dataset.num_node_features, args, class_names)
        self.dataset_class = dataset_class
        self.dataset_kwargs = dataset_kwargs

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

        if dense:
            max_nodes = max([d.num_nodes for d in self.dummy_dataset])
            transforms.append(ToDense(num_nodes=max_nodes))

        if len(transforms) > 0:
            kwargs["transform"] = Compose(transforms) if len(transforms) > 1 else transforms[0]
        del self.dummy_dataset

        dataset = self.dataset_class(**self.dataset_kwargs, **kwargs)
        dataset.shuffle()
        return dataset

class TUDatasetWrapper(PyGWrapper):
    def __init__(self, dataset_name: str, args=None, class_names: Optional[List[str]] = None):
        if args is None:
            args = dict(dataset_name=dataset_name)
        super().__init__(TUDataset, dict(root='/tmp', name=dataset_name), args, class_names)

class MutagWrapper(TUDatasetWrapper):
    """
    TODO: add support for edge features
    """
    def __init__(self):
        super().__init__("MUTAG", dict(), ["not mutagenic", "mutagenic"])

class RedditBinaryWrapper(TUDatasetWrapper):
    def __init__(self):
        super().__init__("REDDIT-BINARY", dict())

class EnzymesWrapper(TUDatasetWrapper):
    def __init__(self):
        super().__init__("ENZYMES", dict())