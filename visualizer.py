import matplotlib
import networkx as nx
import numpy as np
import torch_geometric
from matplotlib import pylab, pyplot as plt
import torch
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from torch_geometric.data import DataLoader, Data
from torch_geometric.utils import k_hop_subgraph
from torch_scatter import scatter

from train import main
import custom_logger


class Visualizer():
    def __init__(self, wandb_id: str, data_loader: DataLoader = None):
        train_loader, val_loader, test_loader = self.load_model(wandb_id)
        data_loader = test_loader if data_loader is None else data_loader
        data_loader = data_loader.__class__(data_loader.dataset, batch_size=len(data_loader.dataset), shuffle=False)
        # extract the first (and only) data batch
        for data in data_loader:
            pass
        self.data = data
        self.load_whole_dataset()
        #self.colors = np.array([matplotlib.colors.rgb2hex(c) for c in pylab.get_cmap("tab20").colors])
        self.input_dim : int = self.model.graph_network.pool_blocks[0].input_dim
        self.colors = np.array(
            ['#2c3e50', '#e74c3c', '#27ae60', '#3498db', '#CDDC39', '#f39c12', '#795548', '#8e44ad', '#3F51B5',
             '#7f8c8d', '#e84393', '#607D8B', '#8e44ad', '#009688'])

    def load_model(self, wandb_id: str):
        self.model, self.config, self.dataset_wrapper, train_loader, val_loader, test_loader =\
            main(dict(resume=wandb_id, save_path="models/dummy.pt"), use_wandb=False, num_epochs=0, shuffle=False)
        self.model.eval()
        return train_loader, val_loader, test_loader

    def load_whole_dataset(self):
        self.data.to(custom_logger.device)
        self.out, _, self.concepts, _, self.pool_assignments, self.pool_activations, self.adjs, self.masks,\
            self.input_embeddings = self.model(self.data, collect_info=True)
        self.y_pred = torch.argmax(self.out, dim=1)
        print(f"Accuracy: {100 * torch.sum(self.y_pred == self.data.y.squeeze(-1)) / self.y_pred.shape[0]:.2f}%")

    def plot_clusters(self, K: int):
        # [num_nodes_total, x, y] (PCA or t-SNE)
        coords = TSNE(n_components=2).fit_transform(X=self.x_out_all)
        kmeans = KMeans(n_clusters=K).fit(X=self.x_out_all)
        # [num_nodes_total] with integer values between 0 and K/num_clusters/num_concepts
        clusters = kmeans.labels_

        markers = ["o", "p", "s", "P", "*", "D", "^", "+", "x"]

        fig, ax = plt.subplots()
        for i in range(self.model.output_dim):
            # Note, we could also evaluate how the model classifies (y_pred_all) instead of the ground truth (y_all) for explainability
            ax.scatter(coords[self.y_all_nodes == i, 0], coords[self.y_all_nodes == i, 1],
                       c=self.colors[clusters[self.y_all_nodes == i]],
                       marker=markers[i], s=10)

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
            colors = self.colors[predicted_clusters]
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