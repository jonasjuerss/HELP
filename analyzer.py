import dataclasses
import math
import random
import re
import warnings
from dataclasses import dataclass
from functools import partial
from typing import Optional, Type, List, Sequence, Tuple

import PIL
import matplotlib
import networkx as nx
import numpy as np
import sklearn
import torch_geometric
import wandb
from functorch import vmap
from matplotlib import pylab, pyplot as plt
import torch
from sklearn.base import BaseEstimator
from sklearn.datasets import load_iris
from sklearn.manifold import TSNE
from torch.utils.data import Subset
from torch_geometric.data import Data
from torch_geometric.loader import DenseDataLoader, DataLoader
from torch_geometric.utils import k_hop_subgraph, to_dense_batch, to_dense_adj, to_networkx
from torch_scatter import scatter
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
import networkx.algorithms.isomorphism as iso

from clustering_wrappers import ClusterAlgWrapper, KMeansWrapper, SequentialKMeansMeanShiftWrapper, MeanShiftWrapper
from color_utils import ColorUtils
from custom_net import InferenceInfo, CustomNet
from data_generation.custom_dataset import HierarchicalMotifGraphTemplate
from data_generation.dataset_wrappers import CustomDatasetWrapper
from graphutils import adj_to_edge_index, mask_to_batch, one_hot
from kmeans import KMeans
from plotstyle import set_dim
from poolblocks.poolblock import DenseNoPoolBlock, SparseNoPoolBlock, DiffPoolBlock, MonteCarloBlock, ASAPBlock
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
        """
        Loads data in dense data format, allowing for transparent handling during analysis
        :param data_loader:
        :param load_part:
        :param mode:
        :param load_names:
        :return:
        """
        set_size = math.floor(load_part * len(data_loader.dataset))
        # Avoids indexing issues with concat dataset when load_part == 1 anyway
        if load_part != 1:
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
            res["x"] = torch.full((set_size, max_nodes, num_node_features), float('nan'))
        if "adj" in load_names:
            res["adj"] = torch.zeros(set_size, max_nodes, max_nodes, dtype=torch.int)
        # if "edge_index" in load_names:
        #     res["edge_index"] = torch.zeros(2, self.dataset_wrapper.num_edges_total, dtype=torch.int)
        if "mask" in load_names:
            res["mask"] = torch.zeros(set_size, max_nodes, dtype=torch.bool)
        if "batch" in load_names:
            res["batch"] = None if self.config.dense_data else torch.zeros(0, dtype=torch.long)

        remap_assignments = "info_pooling_assignments" in load_names and \
                            data_loader.batch_size < set_size and\
                            any(isinstance(b, MonteCarloBlock) and \
                                not isinstance(b.cluster_alg, SequentialKMeansMeanShiftWrapper) and\
                                not b.global_clusters
                                for b in self.model.graph_network.pool_blocks)
        if remap_assignments:
            for pb in self.model.graph_network.pool_blocks:
                if isinstance(pb, MonteCarloBlock) and not (isinstance(pb.cluster_alg, KMeansWrapper) or
                                                            isinstance(pb.cluster_alg, MeanShiftWrapper)):
                    raise ValueError("Can only remap KMeans and MeanShift clustering at the moment."
                                     "In particular, the algorithm must be stateless.")
            res["centroids"] = [[] for _ in self.model.graph_network.pool_blocks]
            print("Remapping assignments between batches!")
        else:
            print("NOT remapping assignments between batches!")

        correct = 0
        sample_i = 0

        def _to_cpu(x):
            if x is None:
                return None
            if isinstance(x, tuple):
                return tuple(a.cpu() for a in x)
            return x.cpu()

        with torch.no_grad():
            for batch_i, data in tqdm(enumerate(data_loader), total=math.ceil(set_size / data_loader.batch_size)):
                target = data.y
                if target.ndim == 2:
                    target = target.squeeze(1)
                sample_i_new = sample_i + target.shape[0]
                if "target" in load_names:
                    res["target"][sample_i:sample_i_new] = target

                target = target.to(custom_logger.device)

                if self.config.dense_data:
                    x, adj, mask = data.x, data.adj, data.mask
                elif "x" in load_names or "adj" in load_names or "mask" in load_names:
                    x, mask = to_dense_batch(data.x, data.batch, max_num_nodes=res["x"].shape[1])
                    adj = to_dense_adj(data.edge_index, data.batch, max_num_nodes=res["x"].shape[1])

                if "x" in load_names:
                    res["x"][sample_i:sample_i_new] = x
                if "adj" in load_names:
                    res["adj"][sample_i:sample_i_new] = adj
                if "mask" in load_names:
                    res["mask"][sample_i:sample_i_new] = mask
                if "batch" in load_names and not self.config.dense_data:
                    res["batch"] = torch.cat((res["batch"], data.batch), dim=0)
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

                if remap_assignments:
                    for i, pb in enumerate(self.model.graph_network.pool_blocks):
                        if hasattr(pb, "cluster_alg"):
                            res["centroids"][i].append(pb.cluster_alg.centroids)

                for key in [f.name for f in dataclasses.fields(InferenceInfo)]:
                    if "info_" + key in load_names:
                        # preprocess
                        if not self.config.dense_data:
                            if key == "pooling_activations":
                                pre_fun = lambda e, inf, i: to_dense_batch(e, batch=data.batch\
                                    if i == 0 else inf.all_batch_or_mask[i - 1])[0]
                            elif key == "adjs_or_edge_indices":
                                pre_fun = lambda e, inf, i: to_dense_adj(e, batch=inf.all_batch_or_mask[i])
                            elif key == "all_batch_or_mask":
                                # inefficient but correct and not time critical in post-hoc analysis
                                pre_fun = lambda e, inf, i: to_dense_batch(e, batch=inf.all_batch_or_mask[i])[1]
                            elif key == "input_embeddings":
                                pre_fun = lambda e, inf, i: to_dense_batch(e, batch=inf.all_batch_or_mask[i])[0]
                            else:
                                pre_fun = lambda e, *_: e
                        else:
                            pre_fun = lambda e, *_: e

                        if "info_" + key in res:
                            for i, e in enumerate(getattr(info, key)):
                                if e is not None:
                                    res["info_" + key][i] = self.cat_pad((res["info_" + key][i], pre_fun(e, info, i).cpu()),
                                                                         dim=0, fill_value=-1\
                                            if key == "pooling_assignments" else 0)

                        else:
                            res["info_" + key] = [_to_cpu(pre_fun(e, info, i)) for i, e in enumerate(getattr(info, key))]

                sample_i = sample_i_new
            if remap_assignments:
                for pool_step, all_centroids in enumerate(res["centroids"]):
                    if not all_centroids:
                        continue
                    cluster_alg = self.model.graph_network.pool_blocks[pool_step].cluster_alg
                    cluster_kwargs = cluster_alg.kwargs
                    cluster_kwargs["kmeans_threshold"] = 0
                    cluster_alg = cluster_alg.__class__(**cluster_kwargs)
                    cluster_alg.kmeans_threshold = 0
                    cluster_alg = cluster_alg.fit_copy(torch.cat(all_centroids, dim=0))
                    # Efficiency is not too important here:
                    for batch_i, cents in enumerate(all_centroids):
                        new_ass = cluster_alg.predict(cents)
                        res["info_pooling_assignments"][pool_step][batch_i] =\
                            new_ass[res["info_pooling_assignments"][pool_step][batch_i]]
                        if torch.any(torch.unique(new_ass, return_counts=True)[1] > 1):
                            warnings.warn("Different centroids were mapped to the same global cluster!")
        if "centroids" in res and "centroids" not in load_names:
            del res["centroids"]

        if "x" in load_names and res["x"].shape[-1] == 0:
            res["x"] = torch.ones(*res["x"].shape[:-1], 1)
        print(f"Loaded {set_size} {mode} samples. Accuracy {100 * correct / set_size:.2f}%")
        return res # data, out, concepts, info, y_pred

    @staticmethod
    def plot_wandb_tables(node_table: wandb.Table, edge_table: wandb.Table, pool_step: int, graph: int,
                          node_size: int = 300, directed=False, save_path: Optional[str] = None):
        border_color_i = node_table.columns.index("border_color")
        r_i = node_table.columns.index("r")
        g_i = node_table.columns.index("g")
        b_i = node_table.columns.index("b")
        node_i = node_table.columns.index("node_index")
        graph_i_n = node_table.columns.index("graph")
        pool_step_i_n = node_table.columns.index("pool_step")
        label_i = node_table.columns.index("permanent_label")

        source_i = edge_table.columns.index("source")
        target_i = edge_table.columns.index("target")
        graph_i_e = edge_table.columns.index("graph")
        pool_step_i_e = edge_table.columns.index("pool_step")
        edge_index = torch.tensor([[e[source_i], e[target_i]] for e in edge_table.data if
                                   e[graph_i_e] == graph and e[pool_step_i_e] == pool_step]).T
        nodes = [n for n in node_table.data if n[graph_i_n] == graph and n[pool_step_i_n] == pool_step]
        num_nodes = len(nodes)
        assert np.all(np.array([n[node_i] for n in nodes]) == np.arange(num_nodes))

        border_colors = [n[border_color_i] for n in nodes]
        fill_colors = [ColorUtils.rgb2hex(int(n[r_i]), int(n[g_i]), int(n[b_i])) for n in nodes]
        data = Data(edge_index=edge_index, num_nodes=num_nodes)
        g = torch_geometric.utils.to_networkx(data, to_undirected=not directed)
        nx.draw(g, node_color=fill_colors, edgecolors=border_colors, linewidths=6, node_size=node_size,
                labels={i: l for i, l in enumerate(n[label_i] for n in nodes)})
        # pos = nx.spring_layout(g)
        # drawn_nodes = nx.draw_networkx_nodes(g, pos, node_size=node_size)
        # drawn_nodes.set_edgecolor(border_colors)
        # nx.draw_networkx_edges(g, pos)
        if save_path is not None:
            plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
        plt.show()

    def log_graphs(self, samples_per_concept: int = 3):
        test_data = self.load_required_data(self.test_loader, samples_per_concept / len(self.test_loader.dataset),
                                             "test",
                                             ["x", "mask", "adj", "info_pooling_assignments",
                                              "info_pooling_activations", "info_all_batch_or_mask",
                                              "info_adjs_or_edge_indices", "info_node_assignments"])
        return self.general_log_graphs(self.model, samples_per_concept=samples_per_concept, **test_data)

    @staticmethod
    def check_one_hot(x: torch.Tensor):
        if torch.any(torch.max(x, dim=-1)[0] != 1) or torch.any(torch.sum(x, dim=-1) != 1) or torch.any(x < 0):
            raise ValueError("Node features are expected to be one-hot!")

    @staticmethod
    def plot_single_concept_examples(examples: List[nx.Graph], pool_step: int, width_scale: float = 1.0) -> PIL.Image:
        fig, axes = plt.subplots(1, len(examples), figsize=(3 * len(examples) * width_scale, 3))
        if len(examples) == 1:
            axes = [axes]
        for i, ex in enumerate(examples):
            Analyzer.plot_nx_concept_graph(ex, pool_step, ax=axes[i])
            axes[i].grid(False)
        fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        fig.canvas.draw()
        return PIL.Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())

    def count_subgraphs(self, pool_step: int, load_part: float = 1, plot_num_nodes: bool = False,
                        plot_num_subgraphs: bool = True, min_occs_to_store: int = 10,
                        max_neighborhoods_to_store: int = 3, use_k_hop: bool = False, save_path: Optional[str] = None,
                        seed: int = 1, inference_with_train: bool = False, use_only_test: bool = True,
                        horizontal: bool = False, purity_threshold: float = 0.1, merge_concepts: bool = False,
                        max_occs_to_merge: int = 5, min_nodes_to_merge: int = 3, min_nodes_for_legend: int = 10,
                        num_gcexplainer_clusters: Optional[int] = None, num_hops: Optional[int] = None,
                        plot_example_graphs: bool = True, example_scale: float = 0.8, example_width_scale: float = 1.0,
                        padding: float = 0.01):
        """

        :param pool_step:
        :param load_part:
        :param plot_num_nodes:
        :param plot_num_subgraphs:
        :param min_occs_to_store:
        :param max_neighborhoods_to_store:
        :param use_k_hop: If true, the generated subgraph for each node will be its k-hop neighborhood rather than all
        connected nodes with the same assignment. This is the way GCExplainer defines concepts
        :param save_path:
        :param seed:
        :param inference_with_train: If True, includes the train data in the inference. This has the sole purpose of
        deterministically producing the same clustering as the decision trees generated in the concept completeness
        :return:
        """
        # Note that the seed not only makes thing reproducible but also ensures that when running the same analysis for the next pool step, the previous concepts align with the features displayed in the next step
        assert inference_with_train or use_only_test
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        # technically, concepts is only needed for GCN/GCExplainer
        required_fields = ["info_pooling_assignments", "concepts", "mask"]
        if pool_step != 0:
            required_fields += ["info_all_batch_or_mask", "info_adjs_or_edge_indices", "info_node_assignments"]
        else:
            required_fields += ["x", "adj"]  # "mask",

        if inference_with_train:
            data_loader = self.train_loader.__class__(self.train_loader.dataset + self.test_loader.dataset,
                                                      self.train_loader.batch_size)
            assert load_part == 1
        else:
            data_loader = self.test_loader
        test_data = self.load_required_data(data_loader, load_part,
                                            ("train and " if inference_with_train else "") + "test",
                                            required_fields)

        if use_only_test:
            test_size = len(self.test_loader.dataset)

            def cut(t):
                if isinstance(t, list):
                    return [None if e is None else e[:test_size] for e in t]
                elif isinstance(t, torch.Tensor):
                    return t[:test_size]
                elif t is None:
                    return None
                raise ValueError()

            test_data = {k: cut(v) for k, v in test_data.items()}

        if pool_step == 0:
            self.check_one_hot(test_data["x"][test_data["mask"]])
            initial_concepts = torch.argmax(test_data["x"], dim=-1).cpu()
            adj = test_data["adj"].cpu()
            mask = test_data["mask"]
        else:
            # Note that in MonteCarlo the pooling assignments are already deterministic anyway
            initial_concepts = \
                Analyzer.deterministic_node_assignments(self.model, test_data["info_pooling_assignments"],
                                                        test_data["info_node_assignments"])[pool_step - 1].cpu()
            test_data["info_node_assignments"][pool_step - 1].cpu()
            adj = test_data["info_adjs_or_edge_indices"][pool_step - 1].cpu()
            mask = test_data["info_all_batch_or_mask"][pool_step - 1]
        if mask is None:
            mask = torch.ones(*adj.shape[:2], dtype=torch.bool)
        else:
            mask = mask.cpu()
        assignment = Analyzer.deterministic_concept_assignments(self.model, test_data["info_pooling_assignments"],
                                                                [mask], test_data["concepts"], num_gcexplainer_clusters,
                                                                )[pool_step].cpu()
        if num_hops is None:
            num_hops = self.model.graph_network.pool_blocks[pool_step].receptive_field
        adj.diagonal(dim1=-2, dim2=-1).copy_(0)  # Remove self-loops
        checked = torch.logical_not(mask)  # masked nodes count as checked
        # Would easily be parallelizable over samples
        num_concepts = torch.max(assignment).item() + 1
        buckets = {}
        num_subgraphs_total = 0
        concept_counts = torch.zeros(num_concepts, dtype=torch.int)
        for sample in tqdm(range(checked.shape[0])):
            nodes_in_cur_graph = torch.sum(mask[sample]).item()
            edge_index_full, _, _ = adj_to_edge_index(adj[sample], mask[sample])
            # plot_nx_graph(to_networkx(Data(#concept=initial_concepts[sample][mask[sample]],
            #     concept=assignment[sample][mask[sample]], edge_index=edge_index, num_nodes=nodes_in_cur_graph), to_undirected=not self.dataset_wrapper.is_directed, node_attrs=["concept"]))
            # plt.show()
            if use_k_hop:
                edge_index = edge_index_full
            else:
                edge_index = edge_index_full[:,
                             assignment[sample, edge_index_full[0, :]] == assignment[sample, edge_index_full[1, :]]]

            # plot_nx_graph(to_networkx(Data(concept=assignment[sample][mask[sample]], edge_index=edge_index, num_nodes=nodes_in_cur_graph), to_undirected=not self.dataset_wrapper.is_directed, node_attrs=["concept"]))
            # plt.show()
            # break

            for node in range(checked.shape[1]):
                if checked[sample, node]:
                    continue
                num_subgraphs_total += 1
                subset, edge_index_loc, mapping, _ = k_hop_subgraph(node,
                                                                    num_hops if use_k_hop else nodes_in_cur_graph,
                                                                    edge_index,
                                                                    relabel_nodes=True,
                                                                    num_nodes=nodes_in_cur_graph)
                if use_k_hop:
                    in_pool = torch.zeros(subset.shape[0], dtype=torch.bool)
                    in_pool[mapping] = True
                else:
                    checked[sample, subset] = True
                    in_pool = None
                cur_concept = assignment[sample, node]
                concept_counts[cur_concept] += 1

                cur_graph = to_networkx(Data(concept=initial_concepts[sample][subset], edge_index=edge_index_loc,
                                             num_nodes=subset.shape[0], in_pool=in_pool),
                                        to_undirected=not self.dataset_wrapper.is_directed,
                                        node_attrs=["concept"] + (["in_pool"] if use_k_hop else []))
                key = nx.algorithms.graph_hashing.weisfeiler_lehman_graph_hash(cur_graph, node_attr="concept")

                if key in buckets:
                    for other_graph, (occurrences, _) in buckets[key].items():
                        if nx.is_isomorphic(other_graph, cur_graph,
                                            node_match=iso.categorical_node_match("concept", None)):
                            buckets[key][other_graph][0][cur_concept] += 1
                            # If subgraph is relevant (at least 10 occurrences), save some neighbourhoods
                            if min_occs_to_store <= buckets[key][other_graph][0][
                                cur_concept] <= min_occs_to_store + max_neighborhoods_to_store:
                                if use_k_hop:
                                    neighborhood = cur_graph
                                else:
                                    subset_n, edge_index_n, mapping_n, _ = k_hop_subgraph(subset, num_hops,
                                                                                          edge_index_full,
                                                                                          relabel_nodes=True,
                                                                                          num_nodes=nodes_in_cur_graph)
                                    in_pool = torch.zeros(subset_n.shape[0], dtype=torch.bool)
                                    in_pool[mapping_n] = True
                                    neighborhood = to_networkx(
                                        Data(concept=initial_concepts[sample][subset_n], in_pool=in_pool,
                                             edge_index=edge_index_n,
                                             num_nodes=subset_n.shape[0]),
                                        to_undirected=not self.dataset_wrapper.is_directed,
                                        node_attrs=["concept", "in_pool"])
                                buckets[key][other_graph][1][cur_concept].append(neighborhood)
                            break
                    else:
                        buckets[key][cur_graph] = one_hot(cur_concept, num_concepts, dtype=torch.int), [[] for _ in
                                                                                                        range(
                                                                                                            num_concepts)]
                else:
                    buckets[key] = {cur_graph: (
                        one_hot(cur_concept, num_concepts, dtype=torch.int), [[] for _ in range(num_concepts)])}

        subgraphs = []
        for key in buckets:
            for g, (counts, examples) in buckets[key].items():
                subgraphs.append((g, counts, examples))

        subgraphs.sort(key=lambda x: torch.sum(x[1]), reverse=True)

        ############################### Soften chart ####################################
        matcher_type = iso.DiGraphMatcher if self.dataset_wrapper.is_directed else iso.GraphMatcher
        counts_are_nodes = False
        if plot_num_nodes:
            if plot_num_subgraphs:
                raise ValueError("Can't plot number of subgraphs and nodes simultaneously anymore!")
            plot_num_subgraphs = True
            plot_num_nodes = False
            subgraphs = [(sg, counts * sg.number_of_nodes(), examples) for (sg, counts, examples) in subgraphs]
            counts_are_nodes = True
        total_counts = torch.sum(torch.stack([counts for (_, counts, _) in subgraphs]), dim=0)  # TODO remove sanity check
        if merge_concepts:
            for concept in range(num_concepts):
                concept_graphs = [sg for sg in subgraphs if sg[1][concept] > 0]
                concept_graphs.sort(key=lambda x: x[1][concept], reverse=True)
                for i in reversed(range(len(concept_graphs))):
                    superg, superg_counts, superg_neighborhoods = concept_graphs[i]
                    if superg_counts[concept] > max_occs_to_merge:
                        continue
                    for j in reversed(range(i)):
                        subg, subg_counts, subg_neighborhoods = concept_graphs[j]
                        if subg.number_of_nodes() >= min_nodes_to_merge and \
                                matcher_type(superg, subg, node_match=iso.categorical_node_match("concept", None)). \
                                        subgraph_is_isomorphic():
                            subg_counts[concept] += superg_counts[concept]
                            superg_counts[concept] = 0
                            total_counts_new = torch.sum(torch.stack([counts for (_, counts, _) in subgraphs]),
                                                         dim=0)  # TODO remove sanity check
                            if not torch.all(total_counts_new == total_counts):
                                print(total_counts, total_counts_new)
                            break

            subgraphs = [sg for sg in subgraphs if torch.max(sg[1]) > 0]
            # Needs to be sorted again after merging
            subgraphs.sort(key=lambda x: torch.sum(x[1]), reverse=True)
        #################################################################################
        if plot_num_subgraphs or plot_num_nodes:
            if horizontal:
                figsize = (8.27, 11.69)
            else:
                figsize = (20 if plot_num_subgraphs and plot_num_nodes else 10, 6)
            fig, axes = plt.subplots(1, 2 if plot_num_nodes and plot_num_subgraphs else 1, figsize=figsize)
            if not horizontal:
                set_dim(fig)
            if plot_num_subgraphs:
                ax_sub = axes if not plot_num_nodes else axes[0]
                if horizontal:
                    ax_sub.set_xlabel("assigned " + "nodes" if counts_are_nodes else "subgraphs")
                else:
                    ax_sub.set_ylabel("assigned " + "nodes" if counts_are_nodes else "subgraphs")
                subgraph_threshold = 0.01 * num_subgraphs_total
            if plot_num_nodes:
                ax_nodes = axes if not plot_num_subgraphs else axes[1]
                # nodes_threshold = 0.01 * torch.sum(mask)
                if horizontal:
                    ax_nodes.set_xlabel("assigned nodes")
                else:
                    ax_nodes.set_ylabel("assigned nodes")
            bottom = torch.zeros(num_concepts, dtype=torch.int)
            bottom_numnodes = torch.zeros(num_concepts, dtype=torch.int)

            # for ax in axes if plot_num_subgraphs and plot_num_nodes else [axes]:
            # NUM_COLORS = len(subgraphs)
            # cm = plt.get_cmap('nipy_spectral') # https://matplotlib.org/stable/tutorials/colors/colormaps.html
            # colors = [cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)]
            # random.shuffle(colors)
            # colors = ["#41ff89", "#3822ca", "#b1f904", "#8220d4", "#d2ef00", "#000698", "#9fff56", "#bd3bee", "#00d643", "#f322dd", "#069700", "#ff42d1", "#8aff84", "#5c0091", "#62ffa1", "#9a0098", "#73ac00", "#0068f7", "#ffd222", "#0061db", "#fffe73", "#1f005b", "#e8ff8e", "#00247a", "#ffe97c", "#b77bff", "#00aa4e", "#ff52b4", "#01dc9f", "#ed0023", "#00d4f7", "#ff393b", "#01bc83", "#ff3c96", "#007721", "#ef94ff", "#406f00", "#478cff", "#eb7d00", "#009dfc", "#ff7030", "#004396", "#aca000", "#99007d", "#c3ffa6", "#500057", "#a1ffd2", "#da004f", "#02bfd2", "#ad1400", "#52c0ff", "#c84200", "#0081b7", "#c97900", "#00427e", "#ab8500", "#aaa6ff", "#758500", "#ff9ee2", "#017439", "#ff4962", "#007e67", "#c00050", "#d5ffed", "#190028", "#ffe999", "#003560", "#ffa75f", "#004777", "#fdffd5", "#440026", "#dff6ff", "#9a001f", "#afeaff", "#8c2b00", "#a7c8ff", "#4b6100", "#d9baff", "#003210", "#ff6f9c", "#005337", "#a3005a", "#007b87", "#890039", "#f3e0ff", "#33000d", "#ffcdc7", "#00211f", "#ffc1ef", "#5f5800", "#ffb0b9", "#2b0e00", "#ffc2a5", "#005068", "#ff9a7d", "#3f3100", "#ff908b", "#4a1d00", "#5f4600", "#6a3600"]
            # colors = []
            # for cmap in ["tab20", "tab20b", "tab20c"]:
            #     cm = plt.get_cmap(cmap)
            #     colors += [cm(1. * i / 20) for i in range(20)]
            # ax.set_prop_cycle('color', colors)

            total_counts = torch.sum(torch.stack([counts for (_, counts, _) in subgraphs]), dim=0)
            max_count = torch.max(total_counts).item()
            relative_subg_size = round((example_scale * figsize[1] / num_concepts) * (max_count / figsize[0]))
            padding *= relative_subg_size # so it becomes relative to the width
            for i, (g, counts, example_graphs) in enumerate(subgraphs):
                if plot_num_subgraphs:
                    if horizontal:
                        ax_sub.barh(np.arange(num_concepts), counts, left=bottom,
                                    label=f"{i}" if torch.max(counts) >= min_nodes_for_legend else None)
                        if plot_example_graphs:
                            for concept in range(num_concepts):
                                num_examples = min( # 0.01: part of width used as padding
                                    math.floor((counts[concept] - 2 * padding) / (example_width_scale * relative_subg_size)),
                                    len(example_graphs[concept]))
                                if num_examples <= 0:
                                    continue
                                img = Analyzer.plot_single_concept_examples(example_graphs[concept][:num_examples],
                                                                            pool_step, example_width_scale)
                                ax_sub.imshow(img, extent=[bottom[concept] + padding,
                                                           bottom[concept] - padding +\
                                                           relative_subg_size * example_width_scale * num_examples,
                                                           concept - (example_scale / 2),
                                                           concept + (example_scale / 2)],
                                              aspect='auto', zorder=2)

                    else:
                        ax_sub.bar(np.arange(num_concepts), counts, bottom=bottom,
                                   label=f"{i}" if torch.max(counts) >= min_nodes_for_legend else None)
                    bottom += counts
                if plot_num_nodes and not merge_concepts:
                    if horizontal:
                        ax_nodes.barh(np.arange(num_concepts), counts * g.number_of_nodes(), left=bottom_numnodes,
                                      label=f"{i}" if torch.max(
                                          counts) * g.number_of_nodes() >= min_nodes_for_legend else None)
                    else:
                        ax_nodes.bar(np.arange(num_concepts), counts * g.number_of_nodes(), bottom=bottom_numnodes,
                                     label=f"{i}" if torch.max(
                                         counts) * g.number_of_nodes() >= min_nodes_for_legend else None)
                    bottom_numnodes += counts * g.number_of_nodes()

            for ax in axes if plot_num_subgraphs and plot_num_nodes else [axes]:
                if horizontal:
                    ax.set_ylabel("concept")
                    # ax.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0, title="subgraph id")
                    ax.set_yticks(range(num_concepts))
                    ax.set_xlim(0, max_count)
                    ax.set_ylim(-0.5, num_concepts - 0.5)
                else:
                    # ax.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0, title="subgraph id")
                    ax.set_xlabel("concept")
                # ax.spines["top"].set_visible(False)
                # ax.spines["right"].set_visible(False)
                # ax.spines["bottom"].set_visible(False)
                # ax.spines["left"].set_visible(False)
            if save_path:
                fig.savefig(f"{save_path}", bbox_inches='tight', dpi=800)

        pure_concept_counts = torch.zeros_like(concept_counts)
        purity_thresholds = purity_threshold * concept_counts
        for _, counts, _ in subgraphs:
            pure_concept_counts += torch.where(counts > purity_thresholds, counts, 0)

        return subgraphs, None if counts_are_nodes \
            else torch.mean(pure_concept_counts[concept_counts != 0] / concept_counts[concept_counts != 0])

    @staticmethod
    def tuples_to_tensor(inputs: dict, **kwargs):
        res = torch.empty(len(inputs), **kwargs)
        for i, j in inputs.items():
            res[i] = j
        return res

    @staticmethod
    def plot_nx_concept_graph(graph, pool_step: int, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        concepts = Analyzer.tuples_to_tensor(nx.get_node_attributes(graph, "concept"), dtype=torch.long)
        in_pool_dict = nx.get_node_attributes(graph, "in_pool")
        if in_pool_dict:
            alpha = 0.3 + 0.7 * Analyzer.tuples_to_tensor(in_pool_dict)
        else:
            alpha = None

        if pool_step == 0:
            ColorUtils.ensure_min_hex_feature_colors(torch.max(concepts) + 1)
            node_color = ColorUtils.hex_feature_colors[concepts.numpy()]
        else:
            ColorUtils.ensure_min_hex_colors(torch.max(concepts) + 1)
            node_color = ColorUtils.hex_colors[concepts.numpy()]

        pos = nx.spring_layout(graph)
        nx.draw_networkx_nodes(graph, ax=ax, pos=pos, node_color=node_color, node_size=300, alpha=alpha)
        nx.draw_networkx_edges(graph, ax=ax, pos=pos, alpha=0.3)

        if ColorUtils.feature_labels is not None and pool_step == 0:
            labels = {i: ColorUtils.feature_labels[concepts[i]] for i in range(concepts.shape[0])}
        elif pool_step != 0:
            labels = {i: f"{concepts[i]}" for i in range(concepts.shape[0])}
        else:
            labels = None
        if labels is not None:
            nx.draw_networkx_labels(graph, ax=ax, pos=pos, labels=labels)

    @staticmethod
    def plot_extended_concept_examples(subgraphs: List[Tuple[nx.Graph, torch.Tensor, List[List[nx.Graph]]]],
                                       pool_step: int, num_concepts: int = 40, samples_per_concept: int = 3,
                                       filter_subgraphs: Optional[List[int]] = None,
                                       filter_concepts: Optional[List[int]] = None, save_path: Optional[str] = None):
        split_concepts = []
        for subgraph_i, (_, counts, samples) in enumerate(subgraphs):
            for concept_i, count in enumerate(counts):
                if filter_concepts is not None and concept_i not in filter_concepts:
                    continue
                if count > 0 and (filter_subgraphs is None or subgraph_i in filter_subgraphs):
                    split_concepts.append((subgraph_i, concept_i, count, samples[concept_i]))
        split_concepts.sort(key=lambda x: x[2], reverse=True)

        fig, axes = plt.subplots(num_concepts, samples_per_concept, figsize=(5 * samples_per_concept, 5 * num_concepts))
        for i, (subgraph_i, concept_i, _, examples) in enumerate(split_concepts[:num_concepts]):
            axes[i, 0].set_ylabel(f"Concept {concept_i}, subgraph {subgraph_i}")
            for sample_i, example in enumerate(examples[:samples_per_concept]):
                Analyzer.plot_nx_concept_graph(example, pool_step, ax=axes[i, sample_i])
        plt.show()
        if save_path:
            fig.savefig(f"{save_path}", bbox_inches='tight')

    @staticmethod
    def deterministic_concept_assignments(model: CustomNet, info_pooling_assignments: List[torch.Tensor],
                                          masks: List[torch.Tensor],
                                          concepts: Optional[List[torch.Tensor]] = None,
                                          num_gcexplainer_clusters: Optional[int] = None) -> List[torch.Tensor]:
        dummy = model.graph_network.pool_blocks
        res = []
        for i, (block, ass) in enumerate(zip(model.graph_network.pool_blocks, info_pooling_assignments)):
            if block.__class__ == DiffPoolBlock:
                res.append(torch.argmax(ass, dim=-1))
            elif block.__class__ == ASAPBlock:
                # TODO Check again how stuff works for ASAP. I'm sure we can plot the assignments but the meaning of a
                #  particular assignment might change from sample to sample because they are only the top k most
                #  important ones. There might not be a unique number like cluster id or new node id that mean the same
                #  thing across graphs. This is another important drawback that I should mention in my thesis
                #  (put whatever I find out in the thesis directly).
                sparse_assignments = torch.zeros(ass[1].shape[0], dtype=torch.long, device=ass[0].device)
                sparse_assignments[ass[0]] = 1
                res.append(to_dense_batch(sparse_assignments, mask_to_batch(masks[i]),
                                          max_num_nodes=masks[i].shape[-1])[0])
            elif block.__class__ in [MonteCarloBlock]:
                res.append(ass)
            elif block.__class__ == DenseNoPoolBlock:
                if len(model.graph_network.pool_blocks) == 1 and concepts is not None:
                    # GCEXplainer for pure GCN
                    # Note that these are just the predictions of the last GNN layer
                    sparse_assignments, _ = KMeans(num_gcexplainer_clusters).fit_predict(concepts[masks[0]])
                    res.append(to_dense_batch(sparse_assignments, mask_to_batch(masks[0]),
                                              max_num_nodes=concepts.shape[-2])[0])
            elif block.__class__ not in [SparseNoPoolBlock]:
                raise NotImplementedError(f"PoolBlock type not supported {block.__class__}")
        return res

    @staticmethod
    def deterministic_node_assignments(model: CustomNet, det_pooling_assignments: List[torch.Tensor],
                                       info_node_assignments: List[torch.Tensor], check=True) -> List[
        torch.Tensor]:
        res = []
        for i, (block, ass) in enumerate(zip(model.graph_network.pool_blocks, info_node_assignments)):
            if block.__class__ == DiffPoolBlock:
                res.append(torch.arange(block.num_output_nodes, device=ass.device).repeat(ass.shape[0], 1))
            elif block.__class__ == ASAPBlock:
                if i < len(det_pooling_assignments) - 1:
                    res.append(torch.zeros(det_pooling_assignments[i].shape[0], det_pooling_assignments[i + 1].shape[1],
                                           device=det_pooling_assignments[i].device, dtype=torch.long))
                else:
                    res.append(None)
            elif block.__class__ == MonteCarloBlock:
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
                           mask: torch.Tensor, info_pooling_activations: List[torch.Tensor],
                           info_all_batch_or_mask: List[torch.Tensor], info_node_assignments: List[torch.Tensor],
                           info_adjs_or_edge_indices: List[torch.Tensor], info_pooling_assignments: List[torch.Tensor],
                           samples_per_concept: int, epoch: Optional[int] = None):
        # IMPORTANT: Here it is crucial to have batches of the size used during training in the forward pass
        # if using only a single example, some concepts might not be present but we still enforce the same number of
        # clusters
        node_table = wandb.Table(["graph", "pool_step", "node_index", "r", "g", "b", "border_color", "label",
                                  "activations", "permanent_label"])
        edge_table = wandb.Table(["graph", "pool_step", "source", "target"])
        with torch.no_grad():
            # concepts: [batch_size, max_num_nodes_final_layer, embedding_dim_out_final_layer] the node embeddings of the final graph
            pool_activations = [x] + info_pooling_activations
            adjs = [adj] + info_adjs_or_edge_indices
            masks = [mask] + [torch.ones(*a.shape[:2], dtype=torch.bool, device=a.device) if m is None else m
                              for m, a in zip(info_all_batch_or_mask, info_adjs_or_edge_indices)]
            pool_assignments = Analyzer.deterministic_concept_assignments(model, info_pooling_assignments, masks)
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
                    ColorUtils.ensure_min_rgb_feature_colors(torch.max(node_assignment) + 1)
                    # [num_nodes, 3] (intermediate dimensions: num_nodes x num_clusters x 3)
                    feature_colors = ColorUtils.rgb_feature_colors[node_assignment, :]
                    for i, i_old in enumerate(masks[pool_step][graph_i].nonzero().squeeze(1)):
                        node_table.add_data(graph_i, pool_step, i, feature_colors[i, 0].item(),
                                            feature_colors[i, 1].item(), feature_colors[i, 2].item(),
                                            ColorUtils.rgb2hex_tensor(concept_colors[i, :]),
                                            f"Cluster {assignment[i]}",
                                            ", ".join([f"{m.item():.2f}" for m in
                                                       pool_activations[pool_step][graph_i, i_old, :].cpu()]),
                                            "" if pool_step > 0 or ColorUtils.feature_labels is None
                                            else ColorUtils.feature_labels[node_assignment[i]])

                    edge_index, _, _ = adj_to_edge_index(adjs[pool_step][graph_i:graph_i+1, :, :],
                                                         masks[pool_step][graph_i:graph_i+1])
                    for i in range(edge_index.shape[1]):
                        edge_table.add_data(graph_i, pool_step, edge_index[0, i].item(), edge_index[1, i].item())

        custom_logger.log(dict(
            # graphs_table=graphs_table
            node_table=node_table,
            edge_table=edge_table
        ), *({} if epoch is None else {"step": epoch}))
        return node_table, edge_table


    @staticmethod
    def _decision_tree_acc(X_train: np.ndarray, Y_train: np.ndarray, X_test: Optional[np.ndarray] = None,
                           Y_test: Optional[np.ndarray] = None, return_tree: bool = False, **kwargs) ->\
            float | Tuple[float, DecisionTreeClassifier]:
        """

        :param X_train: [num_points, num_features] points to create the tree
        :param Y_train: [num_points] labels to create the tree
        :param X_test: [num_points, num_features] points to measure accuracy (X_train if not given)
        :param Y_test: [num_points] labels to measure accuracy (Y_train if not given)
        :return: accuracy of a decision tree fit to the given data
        """
        tree = DecisionTreeClassifier(**kwargs)
        tree.fit(X_train, Y_train)
        acc = tree.score(X_test if X_test is not None else X_train, Y_test if Y_test is not None else Y_train)
        if return_tree:
            return acc, tree
        return acc

    def calculate_concept_completeness(self, multiset: bool = True, data_part: float = 1.0, seeds: List[int] = [1],
                                       plot_tree: bool = False, max_depth: Optional[int] = None,
                                       save_path: Optional[str] = None, verbose: bool = True,
                                       num_gcexplainer_clusters: Optional[int] = None, **kwargs) ->\
            torch.Tensor:
        """
        Important: For the inputs, this assumes one-hot encoding
        """
        results = []
        for seed in seeds:
            trees = []
            torch.manual_seed(seed)
            np.random.seed(seed)

            train_set_size = math.floor(data_part * len(self.train_loader.dataset))
            test_set_size = math.floor(data_part * len(self.test_loader.dataset))
            data_loader = self.train_loader.__class__(self.train_loader.dataset[:train_set_size] +
                                                      self.test_loader.dataset[:test_set_size],
                                                      self.train_loader.batch_size)

            all_data = self.load_required_data(data_loader, 1, "joint train and test",
                                                 ["x", "target", "info_pooling_assignments", "concepts", "mask"])
            # enumerate_feat = 2 ** torch.arange(all_data["x"].shape[-1],
            #                                    device=all_data["x"].device)[None, None, :]
            # Note that I can't calculate concept completeness for ASAP anyway as I have no mapping from nodes to concepts.
            # so masks is not required in determinisitc_assignemnts


            all_train_assignments = [torch.argmax(all_data["x"][:train_set_size], dim=-1)]

            all_test_assignments = [torch.argmax(all_data["x"][train_set_size:], dim=-1)]

            if True:
                all_ass = Analyzer.deterministic_concept_assignments(self.model, all_data["info_pooling_assignments"],
                                                                     [all_data["mask"]], all_data["concepts"],
                                                                     num_gcexplainer_clusters)
                all_train_assignments += [ass[:train_set_size] for ass in all_ass]
                all_test_assignments += [ass[train_set_size:] for ass in all_ass]
            else:
                all_train_assignments += Analyzer.deterministic_concept_assignments(self.model,
                                                           [None if d is None else d[:train_set_size]
                                                            for d in all_data["info_pooling_assignments"]],
                                                           None, None)
                all_test_assignments += Analyzer.deterministic_concept_assignments(self.model,
                                                                              [None if d is None else d[train_set_size:]
                                                                               for d in all_data["info_pooling_assignments"]],
                                                                              None, None)

            result = []
            for pool_step, (train_assignments, test_assignments) in enumerate(zip(all_train_assignments,
                                                                                  all_test_assignments)):
                if train_assignments is None:
                    continue  # For non-pooling layers
                num_concepts = max(torch.max(train_assignments), torch.max(test_assignments)) + 1
                batched_bincount = vmap(partial(torch.bincount, minlength=num_concepts + 1))

                # [batch_size, num_concepts] (note that assignments contains -1 for masked values)
                multisets_train = batched_bincount(train_assignments + 1)[:, 1:]
                multisets_test = batched_bincount(test_assignments + 1)[:, 1:]

                if not multiset:
                    multisets_train = multisets_train.bool().int()
                    multisets_test = multisets_test.bool().int()

                acc, tree = self._decision_tree_acc(multisets_train.cpu(), all_data["target"][:train_set_size].squeeze(),
                                                    multisets_test.cpu(), all_data["target"][train_set_size:].squeeze(),
                                                    return_tree=True, random_state=seed, max_depth=max_depth)
                result.append(acc)
                trees.append(tree)
            results.append(result)
        # [num_seeds, num_mc_blocks + 1]
        results = torch.tensor(results)
        stds, means = torch.std_mean(results, dim=0)
        if verbose:
            for i in range(stds.shape[0]):
                print(f"{100*means[i]:.2f}%+-{100*stds[i]:.2f}")
        if plot_tree:
            for i, tree in enumerate(trees):
                fig, ax = plt.subplots(figsize=(48, 12))
                sklearn.tree.plot_tree(tree, ax=ax, proportion=True,
                                       class_names=self.dataset_wrapper.class_names, fontsize=11, impurity=True)

                # up = False
                def replace_text(obj):
                    # nonlocal up
                    if type(obj) == matplotlib.text.Annotation:
                        txt = obj.get_text()
                        txt = re.sub("value[^$]*class", "class", txt)
                        txt = re.sub(":", ":\n", txt)
                        obj.set_text(txt)
                        # obj.set(y=obj.xy[1]+0.02)
                    # up = not up
                    return obj

                # Remove "value" https://stackoverflow.com/questions/70553185/how-to-plot-tree-without-showing-samples-and-value-in-random-forest
                ax.properties()['children'] = [replace_text(i) for i in ax.properties()['children']]

                if save_path is not None:
                    fig.savefig(f"{save_path}_{i}.pdf")
                plt.show()
        return results

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