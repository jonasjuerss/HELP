import dataclasses
import math
import warnings
from dataclasses import dataclass
from functools import partial
from typing import Optional, Type, List, Sequence

import matplotlib
import networkx as nx
import numpy as np
import torch_geometric
import wandb
from functorch import vmap
from matplotlib import pylab, pyplot as plt
import torch
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from torch.utils.data import Subset
from torch_geometric.data import Data
from torch_geometric.loader import DenseDataLoader, DataLoader
from torch_geometric.utils import k_hop_subgraph
from torch_scatter import scatter
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

from clustering_wrappers import ClusterAlgWrapper
from color_utils import ColorUtils
from custom_net import InferenceInfo, CustomNet
from data_generation.custom_dataset import HierarchicalMotifGraphTemplate
from data_generation.dataset_wrappers import CustomDatasetWrapper
from graphutils import adj_to_edge_index
from poolblocks.poolblock import DenseNoPoolBlock, SparseNoPoolBlock, DiffPoolBlock, MonteCarloBlock
from train import main
import custom_logger



class Analyzer():
    def __init__(self, wandb_id: str, resume_last: bool, train_loader: DataLoader = None,
                 test_loader: DataLoader = None, val_loader: DataLoader = None, use_wandb: bool = False, **overwrite_args):
        _train_loader, _val_loader, _test_loader = self.load_model(wandb_id, use_wandb=use_wandb,
                                                                   resume_last=resume_last, **overwrite_args)
        self.train_loader = _train_loader if train_loader is None else train_loader
        self.test_loader = _test_loader if test_loader is None else test_loader
        self.val_loader = _val_loader if val_loader is None else val_loader
        self.using_wandb = False
        self.step = 0

        # self.train_data, self.train_out, self.train_concepts, train_info, self.train_y_pred = \
        #     self.load_whole_dataset(_train_loader if train_loader is None else train_loader, load_part, "train")
        # self.test_data, self.test_out, self.test_concepts, test_info, self.test_y_pred = \
        #     self.load_whole_dataset(_test_loader if test_loader is None else test_loader, load_part, "test")
        # self.train_info: InferenceInfo = train_info
        # self.test_info: InferenceInfo = test_info

        #hex_colors = np.array([matplotlib.colors.rgb2hex(c) for c in pylab.get_cmap("tab20").colors])
        self.input_dim : int = self.model.graph_network.pool_blocks[0].input_dim
        if "batch_size" in overwrite_args:
            warnings.warn("Overwriting the batch size can have a significant impact on the clustering and therefore "
                          "the inference behaviour!")

    def load_model(self, wandb_id: str, use_wandb: bool, **overwrite_args):
        self.model, self.config, self.dataset_wrapper, train_loader, val_loader, test_loader =\
            main(dict(resume=wandb_id, save_path="models/dummy.pt"), use_wandb=use_wandb, num_epochs=0, shuffle=False,
                 wandb_tags=["Analyzer"], **overwrite_args)
        self.model.eval()
        return train_loader, val_loader, test_loader

    def load_whole_dataset(self, data_loader: DataLoader, load_part: float, mode: str):
        num_samples = math.floor(load_part * len(data_loader.dataset))
        data_loader = data_loader.__class__(data_loader.dataset,
                                            batch_size=num_samples,
                                            shuffle=False)
        # extract the first (and only) data batch
        for data in data_loader:
            pass
        data.to(custom_logger.device)
        with torch.no_grad():
            out, _, concepts, _, info = self.model(data, collect_info=True)
        y_pred = torch.argmax(out, dim=1)
        print(f"Loaded {num_samples} {mode} samples. Accuracy {100 * torch.sum(y_pred == data.y.squeeze(-1)) / y_pred.shape[0]:.2f}%")
        return data, out, concepts, info, y_pred

    @staticmethod
    def cat_pad(inputs: Sequence[torch.Tensor], dim: int, fill_value: float = 0):
        # Alternatively could use: https://github.com/pytorch/pytorch/issues/10978
        res_shape = tuple(max(inp.shape[i] for inp in inputs) for i in range(dim)) + \
                    (sum([inp.shape[dim] for inp in inputs]),) + \
                    tuple(max(inp.shape[i] for inp in inputs) for i in range(dim + 1, inputs[0].ndim))
        res = torch.full(res_shape, fill_value=fill_value, device=inputs[0].device, dtype=inputs[0].dtype)
        start = 0
        for i, inp in enumerate(inputs):
            res[tuple(slice(0, inp.shape[j]) for j in range(dim)) + (slice(start, start + inp.shape[dim]),)
                + tuple(slice(0, inp.shape[j]) for j in range(dim + 1, inputs[0].ndim))] = inp
            start += inp.shape[dim]
        return res

    def load_required_data(self, data_loader: DataLoader, load_part: float, mode: str, load_names: List[str]):
        set_size = math.floor(load_part * len(data_loader.dataset))
        data_loader = data_loader.__class__(data_loader.dataset[:set_size], data_loader.batch_size)
        num_classes = self.dataset_wrapper.num_classes
        max_nodes = self.dataset_wrapper.max_nodes_per_graph
        num_node_features = self.dataset_wrapper.num_node_features
        res = dict()
        if "y_pred" in load_names:
            res["y_pred"] = torch.empty(set_size, dtype=torch.long)
        if "target" in load_names:
            res["target"] = torch.empty(set_size, dtype=torch.long)
        if "out" in load_names:
            res["out"] = torch.empty(set_size, num_classes)
        if "x" in load_names:
            if self.config.dense_data:
                res["x"] = torch.full((set_size, max_nodes, num_node_features), float('nan'))
            else:
                res["x"] = torch.empty(self.dataset_wrapper.num_nodes_total, num_node_features)
        if "adj" in load_names:
            res["adj"] = torch.zeros(set_size, max_nodes, max_nodes, dtype=torch.int)
        if "edge_index" in load_names:
            res["edge_index"] = torch.zeros(2, self.dataset_wrapper.num_edges_total, dtype=torch.int)
        if "mask" in load_names:
            res["mask"] = torch.zeros(set_size, max_nodes, dtype=torch.bool)
        if "batch" in load_names:
            res["batch"] = torch.zeros(set_size, dtype=torch.long)

        sample_i = 0
        node_i = 0
        edge_i = 0
        correct = 0
        with torch.no_grad():
            for batch_i, data in tqdm(enumerate(data_loader), total=math.ceil(set_size / data_loader.batch_size)):
                target = data.y
                if self.config.dense_data:
                    target = target.squeeze(1)
                sample_i_new = sample_i + target.shape[0]
                if "target" in load_names:
                    res["target"][sample_i:sample_i_new] = target

                target = target.to(custom_logger.device)

                if self.config.dense_data:
                    if "x" in load_names:
                        res["x"][sample_i:sample_i_new] = data.x
                    if "adj" in load_names:
                        res["adj"][sample_i:sample_i_new] = data.adj
                    if "mask" in load_names:
                        res["mask"][sample_i:sample_i_new] = data.mask
                else:
                    if "x" in load_names:
                        res["x"][node_i:node_i + data.x.shape[1]] = data.x
                    if "edge_index" in load_names:
                        res["edge_index"][:, edge_i:edge_i + data.edge_index.shape[1]] = data.edge_index
                    if "batch" in load_names:
                        res["batch"][node_i:node_i + data.x.shape[1]] = data.batch
                    node_i += data.x.shape[0]
                    edge_i += data.edge_index.shape[1]

                data.to(custom_logger.device)
                out, _, concepts, _, info = self.model(data, collect_info=True)
                y_pred = torch.argmax(out, dim=1)

                correct += torch.sum(y_pred == target).item()

                if "y_pred" in load_names:
                    res["y_pred"][sample_i:sample_i_new] = y_pred.cpu()
                if "out" in load_names:
                    res["out"][sample_i:sample_i_new] = out.cpu()

                if "concepts" in load_names:
                    if "concepts" in res:
                        res["concepts"] = torch.cat((res["concepts"], concepts), dim=0)
                    else:
                        res["concepts"] = concepts

                for key in [f.name for f in dataclasses.fields(InferenceInfo)]:
                    # TODO support sparse
                    if "info_" + key in load_names:
                        if "info_" + key in res:
                            for i, e in enumerate(getattr(info, key)):
                                if e is not None:
                                    res["info_" + key][i] = self.cat_pad((res["info_" + key][i], e.cpu()), dim=0,
                                                                         fill_value=-1 if key == "pooling_assignments" else 0)

                        else:
                            res["info_" + key] = [None if e is None else e.cpu() for e in getattr(info, key)]

                sample_i = sample_i_new

        print(f"Loaded {set_size} {mode} samples. Accuracy {100 * correct / set_size:.2f}%")
        return res # data, out, concepts, info, y_pred

    def log_graphs(self, samples_per_concept: int = 3):
        train_data = self.load_required_data(self.train_loader, samples_per_concept / len(self.train_loader.dataset),
                                             "train",
                                             ["x", "mask", "adj", "info_pooling_assignments",
                                              "info_pooling_activations", "info_all_batch_or_mask",
                                              "info_adjs_or_edge_indices", "info_node_assignments"])
        self.general_log_graphs(self.model, samples_per_concept=samples_per_concept, **train_data)

    @staticmethod
    def deterministic_concept_assignments(model: CustomNet, info_pooling_assignments: torch.Tensor) -> List[torch.Tensor]:
        res = []
        for i, ass in enumerate(info_pooling_assignments):
            if model.graph_network.pool_blocks[i].__class__ == DiffPoolBlock:
                res.append(torch.argmax(ass, dim=-1))
            elif model.graph_network.pool_blocks[i].__class__ in [MonteCarloBlock]:
                res.append(ass)
            elif model.graph_network.pool_blocks[i].__class__ not in [SparseNoPoolBlock, DenseNoPoolBlock]:
                raise NotImplementedError(f"PoolBlock type not supported {model.graph_network.pool_blocks[i].__class__}")
        return res

    @staticmethod
    def deterministic_node_assignments(model: CustomNet, det_pooling_assignments: torch.Tensor,
                                       info_node_assignments: torch.Tensor, check=True) -> List[
        torch.Tensor]:
        res = []
        for i, (block, ass) in enumerate(zip(model.graph_network.pool_blocks, info_node_assignments)):
            if block.__class__ == DiffPoolBlock:
                res.append(torch.arange(block.num_output_nodes, device=ass.device).repeat(ass.shape[0], 1))
            elif block.__class__ in [MonteCarloBlock]:
                scat = scatter(det_pooling_assignments[i], ass, reduce="min", dim=-1)[:, 1:]
                res.append(scat)
                if check:
                    assert torch.allclose(scat, scatter(det_pooling_assignments[i], ass,
                                                        reduce="max", dim=-1)[:, 1:])
            elif block.__class__ not in [SparseNoPoolBlock, DenseNoPoolBlock]:
                raise NotImplementedError(
                    f"PoolBlock type not supported {model.graph_network.pool_blocks[i].__class__}")
        return res

    @staticmethod
    def general_log_graphs(model: CustomNet, x: torch.Tensor, adj: torch.Tensor,
                                      mask: torch.Tensor, info_pooling_activations: torch.Tensor,
                                      info_all_batch_or_mask: torch.Tensor, info_node_assignments: torch.Tensor,
                                      info_adjs_or_edge_indices: torch.Tensor, info_pooling_assignments: torch.Tensor,
                                      samples_per_concept: int, epoch: Optional[int] = None):
        # IMPORTANT: Here it is crucial to have batches of the size used during training in the forward pass
        # if using only a single example, some concepts might not be present but we still enforce the same number of
        # clusters
        node_table = wandb.Table(["graph", "pool_step", "node_index", "r", "g", "b", "border_color", "label", "activations"])
        edge_table = wandb.Table(["graph", "pool_step", "source", "target"])
        with torch.no_grad():
            # concepts: [batch_size, max_num_nodes_final_layer, embedding_dim_out_final_layer] the node embeddings of the final graph
            pool_activations = [x] + info_pooling_activations
            adjs = [adj] + info_adjs_or_edge_indices
            masks = [mask] + [torch.ones(*a.shape[:2], dtype=torch.bool, device=a.device) if m is None else m
                              for m, a in zip(info_all_batch_or_mask, info_adjs_or_edge_indices)]
            pool_assignments = Analyzer.deterministic_concept_assignments(model, info_pooling_assignments)
            node_concepts = [torch.argmax(x, dim=-1)]\
                            + Analyzer.deterministic_node_assignments(model, pool_assignments, info_node_assignments)

            for pool_step in range(len(pool_assignments)):
                for graph_i in range(samples_per_concept):
                    # Calculate concept assignment colors
                    # [num_nodes] (with batch dimension and masked nodes removed)
                    assignment = pool_assignments[pool_step][graph_i][masks[pool_step][graph_i]].detach().cpu()
                    ColorUtils.ensure_min_rgb_colors(torch.max(assignment) + 1)
                    # [num_nodes, 3] (intermediate dimensions: num_nodes x num_clusters x 3)
                    concept_colors = ColorUtils.rgb_colors[assignment, :]

                    node_assignment = node_concepts[pool_step][graph_i][masks[pool_step][graph_i]].detach().cpu()
                    ColorUtils.ensure_min_rgb_colors(torch.max(node_assignment) + 1)
                    # [num_nodes, 3] (intermediate dimensions: num_nodes x num_clusters x 3)
                    feature_colors = ColorUtils.rgb_colors[node_assignment, :]
                    for i, i_old in enumerate(masks[pool_step][graph_i].nonzero().squeeze(1)):
                        node_table.add_data(graph_i, pool_step, i, feature_colors[i, 0].item(),
                                            feature_colors[i, 1].item(), feature_colors[i, 2].item(),
                                            ColorUtils.rgb2hex_tensor(concept_colors[i, :]),
                                            f"Cluster {assignment[i]}",
                                            ", ".join([f"{m.item():.2f}" for m in
                                                       pool_activations[pool_step][graph_i, i_old, :].cpu()]))

                    edge_index, _, _ = adj_to_edge_index(adjs[pool_step][graph_i:graph_i+1, :, :],
                                                         masks[pool_step][graph_i:graph_i+1])
                    for i in range(edge_index.shape[1]):
                        edge_table.add_data(graph_i, pool_step, edge_index[0, i].item(), edge_index[1, i].item())

        custom_logger.log(dict(
            # graphs_table=graphs_table
            node_table=node_table,
            edge_table=edge_table
        ), *({} if epoch is None else {"step": epoch}))


    @staticmethod
    def _decision_tree_acc(X_train: np.ndarray, Y_train: np.ndarray, X_test: Optional[np.ndarray] = None,
                           Y_test: Optional[np.ndarray] = None) -> float:
        """

        :param X_train: [num_points, num_features] points to create the tree
        :param Y_train: [num_points] labels to create the tree
        :param X_test: [num_points, num_features] points to measure accuracy (X_train if not given)
        :param Y_test: [num_points] labels to measure accuracy (Y_train if not given)
        :return: accuracy of a decision tree fit to the given data
        """
        tree = DecisionTreeClassifier()
        tree.fit(X_train, Y_train)
        return tree.score(X_test if X_test is not None else X_train, Y_test if Y_test is not None else Y_train)

    def calculate_concept_completeness(self, multiset: bool = True, data_part: float = 1.0, seeds: List[int] = [0]) ->\
            torch.Tensor:
        """
        Important: For the inputs, this assumes one-hot encoding
        """
        results = []
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            train_data = self.load_required_data(self.train_loader, data_part, "train",
                                                 ["x", "target", "info_pooling_assignments"])
            test_data = self.load_required_data(self.test_loader, data_part, "test",
                                                 ["x", "target", "info_pooling_assignments"])
            enumerate_feat = 2 ** torch.arange(train_data["x"].shape[-1],
                                               device=train_data["x"].device)[None, None, :]
            all_train_assignments = [torch.sum(train_data["x"].int() * enumerate_feat, dim=-1)] +\
                                    Analyzer.deterministic_concept_assignments(self.model,
                                                                               train_data["info_pooling_assignments"])
            all_test_assignments = [torch.sum(test_data["x"].int() * enumerate_feat, dim=-1)] + \
                                   Analyzer.deterministic_concept_assignments(self.model,
                                                                              test_data["info_pooling_assignments"])
            result = []
            for pool_step, (train_assignments, test_assignments) in enumerate(zip(all_train_assignments,
                                                                                  all_test_assignments)):
                if train_assignments is None:
                    continue  # For non-MonteCarlo layers
                num_concepts = max(torch.max(train_assignments), torch.max(test_assignments)) + 1
                batched_bincount = vmap(partial(torch.bincount, minlength=num_concepts + 1))

                # [batch_size, num_concepts] (note that assignments contains -1 for masked values)
                multisets_train = batched_bincount(train_assignments + 1)[:, 1:]
                multisets_test = batched_bincount(test_assignments + 1)[:, 1:]

                if not multiset:
                    multisets_train = multisets_train.bool().int()
                    multisets_test = multisets_test.bool().int()

                result.append(self._decision_tree_acc(multisets_train.cpu(), train_data["target"].squeeze(),
                                                      multisets_test.cpu(), test_data["target"].squeeze()))
            results.append(result)
        # [num_seeds, num_mc_blocks + 1]
        results = torch.tensor(results)
        stds, means = torch.std_mean(results, dim=0)
        for i in range(stds.shape[0]):
            print(f"{100*means[i]:.2f}%+-{100*stds[i]:.2f}")
        return results

        # dataset_instance = self.dataset_wrapper.get_dataset(self.config.dense_data, 0)
        # if isinstance(self.dataset_wrapper, CustomDatasetWrapper):
        #     if isinstance(self.dataset_wrapper.sampler, HierarchicalMotifGraphTemplate):
        #         raise NotImplementedError()
        #     else:
        #         raise NotImplementedError()
        # else:
        #     raise NotImplementedError()

    def plot_clusters(self, cluster_alg: Optional[Type[ClusterAlgWrapper]] = None, save_path: Optional[str] = None,
                      dim_reduc: Type[BaseEstimator] = TSNE, data_part: float = 1.0, **cluster_alg_kwargs):
        """
        :param cluster_alg: Type of a clustering algorithm to use. If provided, this will be fitted for every pooling
        layer. Otherwise, only pooling layers that use clustering will be shown (with the cluster algorithm fitted to
        the train as of the last pass).
        :return:
        """
        train_data = self.load_required_data(self.train_loader, data_part, "train",
                                ["mask", "info_pooling_activations", "info_all_batch_or_mask"])
        pool_acts_train = train_data["info_pooling_activations"]
        masks_train = [train_data["mask"]] + train_data["info_all_batch_or_mask"]
        test_data = self.load_required_data(self.test_loader, data_part, "train",
                                            ["mask", "info_pooling_activations", "info_all_batch_or_mask"])
        pool_acts_test = test_data["info_pooling_activations"]
        masks_test = [test_data["mask"]] + test_data["info_all_batch_or_mask"]
        for i in range(len(pool_acts_train)):
            if cluster_alg is None:
                if not hasattr(self.model.graph_network.pool_blocks[i], "cluster_alg"):
                    break
                cluster_alg_used = self.model.graph_network.pool_blocks[i].cluster_alg
            else:
                cluster_alg_used = cluster_alg(**cluster_alg_kwargs)
                cluster_alg_used.fit(pool_acts_train[i][masks_train[i]])

            num_train_nodes = math.ceil(pool_acts_train[i][masks_train[i]].shape[0] * data_part)
            num_nodes = num_train_nodes + math.ceil(pool_acts_test[i][masks_test[i]].shape[0] * data_part)
            all_points = torch.cat(
                (pool_acts_train[i][masks_train[i]].to(custom_logger.device), pool_acts_test[i][masks_test[i]].to(custom_logger.device), cluster_alg_used.centroids),
                dim=0).detach()
            assignments = cluster_alg_used.predict(all_points).cpu()
            all_points = all_points.cpu()

            coords = dim_reduc(n_components=2).fit_transform(X=all_points)
            ColorUtils.ensure_min_hex_colors(torch.max(assignments) + 1)
            fig, ax = plt.subplots()
            # o p s
            ax.scatter(coords[:num_train_nodes, 0], coords[:num_train_nodes, 1],
                       c=ColorUtils.hex_colors[assignments[:num_train_nodes]],
                       marker='s', s=2, label="Train embeddings")
            ax.scatter(coords[num_train_nodes:num_nodes, 0], coords[num_train_nodes:num_nodes, 1],
                       c=ColorUtils.hex_colors[assignments[num_train_nodes:num_nodes]],
                       marker='p', s=2, label="Test embeddings")
            ax.scatter(coords[num_nodes:, 0], coords[num_nodes:, 1],
                       c=ColorUtils.hex_colors[assignments[num_nodes:]],
                       marker='o', s=10, label="Centroids")
            fig.show()
            if save_path is not None:
                fig.savefig(f"{save_path}_{i}.pdf", bbox_inches='tight')

    ###################################################### Legacy ######################################################
    # def plot_clusters(self, K: int):
    #     # [num_nodes_total, x, y] (PCA or t-SNE)
    #     coords = TSNE(n_components=2).fit_transform(X=self.x_out_all)
    #     kmeans = KMeans(n_clusters=K).fit(X=self.x_out_all)
    #     # [num_nodes_total] with integer values between 0 and K/num_clusters/num_concepts
    #     clusters = kmeans.labels_
    #
    #     markers = ["o", "p", "s", "P", "*", "D", "^", "+", "x"]
    #
    #     fig, ax = plt.subplots()
    #     for i in range(self.model.output_dim):
    #         # Note, we could also evaluate how the model classifies (y_pred_all) instead of the ground truth (y_all) for explainability
    #         ax.scatter(coords[self.y_all_nodes == i, 0], coords[self.y_all_nodes == i, 1],
    #                    c=hex_colors[clusters[self.y_all_nodes == i]],
    #                    marker=markers[i], s=10)

    def draw_neighbourhood(self, ax, node_index: int):
        subset, edge_index, mapping, _ = k_hop_subgraph(node_index, self.num_hops, torch.tensor(self.edge_index_all),
                                                        relabel_nodes=True)
        g = torch_geometric.utils.to_networkx(Data(x=subset, edge_index=edge_index), to_undirected=True)
        highlight_mask = np.zeros(subset.shape[0], dtype=int)
        highlight_mask[mapping] = 1
        labels = self.dataset_wrapper.get_node_labels(self.x_all[subset])
        nx.draw(g, ax=ax, node_color=self.dataset_wrapper.get_node_colors(self.annot_all[subset]),
                edgecolors="#8e44ad",
                linewidths=3.0*highlight_mask, labels={i: labels[i] for i in range(labels.shape[0])},
                font_color="whitesmoke")

    def plot_closest_embeddings(self, embeddings: np.ndarray, labels, save_path=None, num_plots = 5):
        """
        For each row in embeddings, plots the num_gnn_layers-hop neighbourhood of the num_plots closest embeddings in x_out_all
        :param embeddings: [num_embeddings, gnn_output_embedding_size]
        """
        fig, axes = plt.subplots(embeddings.shape[0], num_plots, figsize=(15, embeddings.shape[0] * 5))
        for i in range(embeddings.shape[0]):
            # CAUTION: we are using TOP k, so we need a minus in front to find the ones with smallest distance
            # indices of the num_plots nodes that are closest to the center of concept
            _, indices = torch.topk(-torch.norm(torch.tensor(embeddings[i:i+1, :] - self.x_out_all), dim=-1), k=num_plots, dim=0)
            axes[i][0].set_title(f"{labels[i]}", rotation='vertical', x=-0.1, y=0.5)
            for j in range(num_plots):
                self.draw_neighbourhood(axes[i][j], indices[j].item())
        if save_path is not None:
            fig.savefig(save_path, bbox_inches='tight')
        return fig

    def draw_graph(self, sample: int, color_concepts=False, save_path=None, figsize=(7, 7)):
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis('off')
        mask = self.batch_all == sample
        node_indices = np.arange(self.x_all.shape[0])[mask]
        start_index, end_index = node_indices[0].item(), node_indices[-1].item()
        edge_index = self.edge_index_all[:,
                     np.logical_and(self.edge_index_all[0] >= start_index, self.edge_index_all[0] <= end_index)] - start_index
        x = self.x_all[mask]
        g = torch_geometric.utils.to_networkx(Data(x=torch.tensor(x), edge_index=torch.tensor(edge_index)),
                                              to_undirected=True)
        labels = self.dataset_wrapper.get_node_labels(x)
        pred_str = ", ".join([f"{100 * f:.0f}%" for f in np.exp(self.out_all[sample])])
        if color_concepts:
            # IMPORANT: Note that colors of clusters will not be consistent among different samples this way
            _, predicted_clusters = np.unique(np.argmax(self.x_out_all[mask], axis=1), return_inverse=True)
            colors = ColorUtils.hex_colors[predicted_clusters]
        else:
            colors = self.dataset_wrapper.get_node_colors(self.annot_all[mask])
        ax.set_title(
            f"class: {self.y_all[sample]} ({self.dataset_wrapper.class_names[self.y_all[sample].item()]}), prediction: [{pred_str}]")
        nx.draw(g, ax=ax, node_color=colors,
                labels={i: str(i) for i in range(labels.shape[0])}, font_color="whitesmoke")

        if save_path is not None:
            fig.savefig(save_path, bbox_inches='tight')
        return fig

    # Show neighbourhoods of nodes with the most similar embeddings
    def show_nearest(self, sample: int, num_samples_per_concept=5, save_path=None):
        return self.plot_closest_embeddings(self.x_out_all[self.batch_all == sample, :],
                                            [f"Node {i}" for i in range(self.x_out_all[self.batch_all == sample, :].shape[0])],
                                            num_plots=num_samples_per_concept, save_path=save_path)

    # Show neighbourhoods of nodes whos embeddings are closest to the desired one-hot vector
    def show_nearest_discretized(self, sample: int, num_samples_per_concept=5, save_path=None):
        max_indices = np.argmax(self.x_out_all[self.batch_all == sample, :], axis=-1)
        # [num_present_concepts, gnn_sizes[-1]]
        present_concepts = np.eye(self.x_out_all.shape[1])[np.unique(max_indices), :]
        labels = [None for _ in range(self.x_out_all.shape[1])]
        for i in range(max_indices.shape[0]):
            labels[max_indices[i]] = (f"Concept {max_indices[i]}, Nodes:\n" if labels[max_indices[i]] is None else
                                      labels[max_indices[i]] + ", ") + str(i)

        return self.plot_closest_embeddings(present_concepts, [l for l in labels if l is not None],
                                            num_plots=num_samples_per_concept, save_path=save_path)

    def plot_category_histogram(self, save_path=None):
        counts = np.bincount(np.argmax(self.x_out_all, axis=1))
        fig, ax = plt.subplots()
        ax.bar(np.arange(counts.shape[0]), counts)
        if save_path is not None:
            fig.savefig(save_path, bbox_inches='tight')
        return fig

    def calculate_average_activation_shannon_entropy(self):
        nonzero_logs = np.log2(self.x_out_all, out=np.zeros_like(self.x_out_all), where=(self.x_out_all != 0))
        return np.mean(np.sum(-self.x_out_all * nonzero_logs, axis=1))

    def _scatter_mean_and_std(self, values, indices):
        # [num_nodes, k]
        values = torch.tensor(values)
        # [num_nodes]
        indices = torch.tensor(indices)
        # [num_unique_indices, k]
        mean = scatter(values, indices, dim=0, reduce="mean")
        squared_diff = torch.square(values - mean[indices])
        squared_diff_sums = scatter(squared_diff, indices, dim=0, reduce="sum")
        n = scatter(torch.ones_like(values), indices, dim=0, reduce="sum")
        std = torch.sqrt(squared_diff_sums / (n - 1))
        return mean.detach().numpy(), std.detach().numpy()


    def plot_average_theta_and_h(self, save_path=None):
        fig, ax = plt.subplots()
        concepts = torch.tensor(np.argmax(self.x_out_all, axis=1))

        # [num_concepts, num_classes], [num_concepts, num_classes]
        mean_theta, std_theta = self._scatter_mean_and_std(self.theta_all, concepts)
        # [num_concepts, 1], [num_concepts, 1]
        mean_h, std_h = self._scatter_mean_and_std(self.h_all, concepts)

        ax.errorbar(np.arange(mean_h.shape[0]), mean_h, std_h, linestyle='None')

        if save_path is not None:
            fig.savefig(save_path, bbox_inches='tight')
        return fig
    def plot_dropout_effect(self, save_path=None):
        fig, ax = plt.subplots()
        concepts = torch.tensor(np.argmax(self.x_out_all, axis=1))
        scatter()

        if save_path is not None:
            fig.savefig(save_path, bbox_inches='tight')
        return fig