import abc
from argparse import Namespace
from typing import Union, List, Sequence

import torch
from torch import Tensor
from torch_explain.logic.nn import entropy
from torch_explain.nn import EntropyLinear
from torch_explain.nn.functional import entropy_logic_loss
from torch_geometric.loader import DataLoader

from custom_logger import log
from graph_pooling_network import GraphPoolingNetwork


class Classifier(torch.nn.Module, abc.ABC):
    def __init__(self, input_size: Sequence[int], num_classes: int, device, args: Namespace):
        """
        :param num_output_nodes: input shape of the layer. For example [features_per_node] (merge: sum/avg/...),
        [features_per_node * num_output_nodes] (merge: flatten) or [num_output_nodes/None, features_per_node]
        (merge: none). Note that the None here indicates an unknown number of output nodes which is useful for layers on
        sets like DCR.
        :param num_classes: The number of classes to predict
        :param args: Commandline arguments for additional things like certain loss weights (e.g. entropy)
        """
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.device = device

    @abc.abstractmethod
    def forward(self, concepts: Tensor) -> Tensor:
        """
        :param concepts: <ul>
            <li>either: [batch_size, num_output_nodes, features_per_output_node] (for output_layer_merge="none")</li> TODO change
            <li>or: [batch_size, features_per_output_node] (for output_layer_merge="max", "sum")</li>
            <li>or: [batch_size, num_output_nodes * features_per_output_node] (for output_layer_merge="flatten")</li></ul>
            Where the last 2 cases can be handled transparently as long as the number of nodes in the last graph is
            constant
        :return: [batch_size, num_classes]
        """
        pass
    def custom_losses(self, batch_size: int) -> Union[Tensor, float]:
        """
        :return: The weighted (sum of) loss(es) to add. Each of them can also be logged to wandb.
        """
        return 0

    def log_custom_losses(self, mode: str, epoch: int, dataset_length: int):
        """
        Logs accumulated losses and resets them to 0
        :param mode: "test" or "train"
        :param epoch:
        :param dataset_length:
        :return:
        """
        pass


    def explain(self, graph_model: GraphPoolingNetwork, train_loader: DataLoader, test_loader, class_names: List[str])\
            -> Union[None]:
        return None

class DenseClassifier(Classifier):
    """
    Concatenates all output node features and feeds them to a dense layer
    """
    def __init__(self, input_size: Sequence[int], num_classes: int, device, args: Namespace):
        super().__init__(input_size, num_classes, device, args)
        if len(input_size) != 1:
            raise ValueError(f"{self.__class__.__name__} currently does not support more than 1 input dimension "
                             f"({input_size} given)!")
        self.layer = torch.nn.Linear(*input_size, num_classes)

    def forward(self, concepts: Tensor) -> Tensor:
        return self.layer(concepts)


class EntropyClassifier(Classifier):
    """
    Concatenates all output node features and feeds them to an entropy layer
    """
    def __init__(self, input_size: Sequence[int], num_classes: int, device, args: Namespace):
        super().__init__(input_size, num_classes, device, args)
        if len(input_size) != 1:
            raise ValueError(f"{self.__class__.__name__} currently does not support more than 1 input dimension "
                             f"({input_size} given)!")
        self.entropy_loss_weight = args.entropy_loss_weight
        self.sum_entropy_loss = 0
        self.layer = torch.nn.Sequential(
            EntropyLinear(input_size[0], 1, n_classes=num_classes),
            torch.nn.Flatten(1))

    def forward(self, concepts: Tensor) -> Tensor:
        return self.layer(concepts)

    def explain(self, graph_model: GraphPoolingNetwork, train_loader: DataLoader, test_loader, class_names: List[str]):
        return # TODO fix device errors | TODO rethink explanation now that self.prep is moved to custom_net
        # determining the size in advance would be more efficient than stacking
        # [num_samples, gnn_output_size]
        xs = torch.empty((0, self.num_output_nodes), device=self.device)
        # [num_samples]
        ys = torch.empty((0, ), device=self.device, dtype=torch.long)
        train_samples = 0
        for data in train_loader:
            # data.x: [batch_size, nodes_per_graph, input_feature_size/num_colors]
            # data.y: [batch_size, 1]
            # concepts:
            concepts, _, _, _ = graph_model(data.to(self.device))
            xs = torch.cat((xs, self.prep(concepts)), dim=0)
            ys = torch.cat((ys, data.y.squeeze(1)), dim=0)
            train_samples += data.y.shape[0]

        for data in test_loader:
            # data.x: [batch_size, nodes_per_graph, input_feature_size/num_colors]
            # data.y: [batch_size, 1]
            concepts, _, _ = graph_model(data.to(self.device))
            xs = torch.cat((xs, self.prep(concepts)), dim=0)
            ys = torch.cat((ys, data.y.squeeze(1)), dim=0)

        train_mask = torch.arange(train_samples, dtype=torch.long)
        test_mask = torch.arange(ys.shape[0] - train_samples, dtype=torch.long) + train_samples
        concept_names = [f"c{i}" for i in range(xs.shape[1])]
        ys = torch.nn.functional.one_hot(ys)
        explanations, local_exp = entropy.explain_classes(self.layer, xs, ys.cpu(), train_mask, test_mask,
                                                          concept_names=concept_names, class_names=class_names)
        print(explanations)
        print("----------")
        print(local_exp)
        print("##########")

    def custom_losses(self, batch_size: int) -> Tensor:
        entropy_loss = entropy_logic_loss(self.layer)
        self.sum_entropy_loss += batch_size * entropy_loss
        return self.entropy_loss_weight * entropy_loss

    def log_custom_losses(self, mode: str, epoch: int, dataset_length: int) -> None:
        log({f"{mode}_entropy_loss": self.sum_entropy_loss / dataset_length},
            step=epoch, commit=False)
        self.sum_entropy_loss = 0

class DCRClassifier(Classifier):

    def __init__(self, num_output_nodes: Union[int, None], features_per_output_node: Union[int, None],
                 num_classes: int, device, args: Namespace):
        super().__init__(num_output_nodes, features_per_output_node, num_classes, device, args)
        pass

    def forward(self, concepts: Tensor) -> Tensor:
        pass

    def custom_losses(self, batch_size: int) -> Tensor:
        pass

    def log_custom_losses(self, mode: str, epoch: int, dataset_length: int) -> None:
        pass

    def explain(self, graph_model: GraphPoolingNetwork, train_loader: DataLoader, test_loader,
                class_names: List[str]) -> Union[None]:
        pass


def from_name(network_name: str):
    try:
        return globals()[network_name]
    except KeyError:
        raise ValueError(f"Unknown classifier name: {network_name}")