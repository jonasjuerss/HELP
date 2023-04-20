import argparse
from contextlib import nullcontext
from typing import Tuple, Optional

import numpy as np
import torch
import wandb
from torch.distributions import Categorical, Normal
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch_scatter import scatter
from tqdm import tqdm

import custom_logger
from blackbox_backprop import BlackBoxFun, BlackBoxModule
from clustering_wrappers import ClusterAlgWrapper, KMeansWrapper, MeanShiftWrapper


class ClusterDataset():
    def __init__(self, num_categories: int, num_centroids_per_category: int, num_points: int, input_size: int,
                 noise_std: float):
        self.num_classes = 2 ** min(num_categories, num_points) - 1
        self.num_points = num_points
        self.num_categories = num_categories
        self.num_centroids_per_category = num_centroids_per_category
        num_centroids = num_categories * num_centroids_per_category
        # [num_centroids, input_size]
        self.centroids = torch.rand(num_categories * num_centroids_per_category, input_size)
        self.centroid_distr = Categorical(torch.ones(num_centroids))
        self.noise_distr = Normal(torch.tensor(0, dtype=torch.float), torch.tensor(noise_std, dtype=torch.float))
        if self.num_categories <= self.num_points:
            self.class_names = (torch.arange(self.num_classes)[:, None] + 1).bitwise_and(2 ** torch.arange(num_categories))[
                               None, :].ne(0).byte().squeeze(0)
        else:
            raise NotImplementedError()
        self.class_names = ["".join([f"{i}" for i in class_id]) for class_id in self.class_names]

    def sample(self) -> dict:
        # [num_points]
        centroid_ids = self.centroid_distr.sample((self.num_points,))
        # [num_points, input_size]
        points = self.centroids[centroid_ids]
        points += self.noise_distr.sample(points.shape)
        if self.num_categories <= self.num_points:
            y = torch.sum(2 ** torch.unique((centroid_ids // self.num_centroids_per_category))) - 1
        else:
            raise NotImplementedError()
        return dict(x=points, y=y, categories=centroid_ids // self.num_centroids_per_category)


def perform_clustering(x: torch.Tensor, cluster_alg: ClusterAlgWrapper, second_part: torch.nn.Sequential):
    """

    :param x: [batch_size, num_points, intermediate_dim]
    :return:
        clusters: [num_clusters_total, intermediate_dim]
        assignments: [num_clusters_total]
        batch: [num_clusters_total]
    """
    batch_size = x.shape[0]
    # [batch_size * num_points]
    batch = torch.arange(batch_size, device=custom_logger.device).repeat_interleave(x.shape[1])
    # [batch_size * num_points, intermediate_dim]
    x = x.flatten(0, 1)
    # [batch_size * num_points] with values in {0, num_clusters}
    assignments = cluster_alg.fit_predict(x)
    # [batch_size * num_points] mapping points to the same number iff. they are in the same cluster and the same sample
    _, per_sample_assignments = torch.unique(assignments + batch * batch_size, return_inverse=True)
    # [num_clusters_total, intermediate_dim]
    x = scatter(x, per_sample_assignments, reduce="mean", dim=0)
    _, batch = torch.unique_consecutive(scatter(batch, per_sample_assignments, reduce="min", dim=0),
                                        return_inverse=True)

    # [num_clusters_total, pool_dim]
    x = second_part(x)
    # [batch_size, pool_dim]
    x = scatter(x, batch, dim=0, reduce="sum")

    return x, assignments


class BlackboxClusterModule(BlackBoxModule):
    def __init__(self, num_samples: int, noise_std: float, cluster_alg: ClusterAlgWrapper,
                 second_part: torch.nn.Module, transperency: float):
        super().__init__(num_samples,
                         Normal(torch.tensor(0, dtype=torch.float, device=custom_logger.device),
                                torch.tensor(noise_std, dtype=torch.float, device=custom_logger.device)), 2,
                         transperency)
        self.cluster_alg = cluster_alg
        self.second_part = second_part

    def hard_fn(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        # [batch_size * num_points]
        batch = torch.arange(batch_size, device=custom_logger.device).repeat_interleave(x.shape[1])
        # [batch_size * num_points, intermediate_dim]
        x = x.flatten(0, 1)
        # [batch_size * num_points] with values in {0, num_clusters}
        assignments = self.cluster_alg.fit_predict(x)
        # [batch_size * num_points] mapping points to the same number iff. they are in the same cluster and the same sample
        _, per_sample_assignments = torch.unique(assignments + batch * batch_size, return_inverse=True)
        # [num_clusters_total, intermediate_dim]
        x = scatter(x, per_sample_assignments, reduce="mean", dim=0)
        batch = scatter(batch, per_sample_assignments, reduce="min", dim=0)
        return x, assignments, batch

    def postprocess(self, x: torch.Tensor, assignments: torch.Tensor, batch: torch.Tensor) -> \
            Tuple[torch.Tensor, torch.Tensor]:
        # [num_clusters_total, pool_dim]
        x = self.second_part(x)
        # [batch_size, pool_dim]
        x_pooled = scatter(x, batch, dim=0, reduce="sum")
        return x_pooled, assignments

class ClusterModule(torch.nn.Module):
    def __init__(self, input_dim: int, intermediate_dim: int, pool_dim: int, output_dim: int,
                 cluster_alg: ClusterAlgWrapper, num_samples: int, noise_std: float, transperency: float):
        super().__init__()
        self.first_part = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, intermediate_dim)
        )
        second_part = torch.nn.Sequential(
            torch.nn.Linear(intermediate_dim, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, pool_dim)
        )
        self.third_part = torch.nn.Sequential(
            torch.nn.Linear(pool_dim, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, output_dim)
        )
        self.blackbox = BlackboxClusterModule(num_samples, noise_std, cluster_alg, second_part, transperency)
        self._plot_next_embeddings = 0
        self.colors = np.array(['#f44336', '#9c27b0', '#3f51b5', '#03a9f4', '#009688', '#8bc34a', '#ffeb3b',
                                '#ff9800', '#795548', '#607d8b', '#e91e63', '#673ab7', '#2196f3', '#00bcd4',
                                '#4caf50', '#cddc39', '#ffc107', '#ff5722', '#9e9e9e'])

    def forward(self, x: torch.Tensor, **logging_kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        :param x: [batch_size, num_points, input_dim]
        :param logging_kwargs: arguments that will be passed to the logging if it is called
        :return:
            classifications: [batch_size, output_dim]
            assignments: [batch_size * num_points]
        """
        # [batch_size, num_points, intermediate_dim]
        x_coords = self.first_part(x)

        # x: [num_clusters_total, intermediate_dim], assignments: [batch_size * num_points]
        x_pooled, assignments = self.blackbox(x_coords)
        # [batch_size, output_dim]
        y_pred = torch.log_softmax(self.third_part(x_pooled), dim=-1)
        if self._plot_next_embeddings > 0:
            self.plot_embeddings(x, x_coords, assignments, y_pred, **logging_kwargs)
        return y_pred, assignments

    def plot_next_embeddings(self, n: int):
        self._plot_next_embeddings = n

    def plot_embeddings(self, x_in: torch.Tensor, x_coords: torch.Tensor, assignments: torch.Tensor, y_pred: torch.Tensor, y: torch.Tensor,
                        categories: torch.Tensor, dataset: ClusterDataset, epoch: int):
        """

        :param x_coords: [batch_size, num_points, intermediate_dim]
        :param assignments: [batch_size * num_points]
        :param y_pred: [batch_size, output_dim]
        :param y: [batch_size]
        :param categories: [batch_size, num_categories] (always on cpu for efficiency as often not needed)
        :param dataset:
        :param epoch:
        :return:
        """
        # [batch_size, num_points]
        assignments = assignments.reshape(x_coords.shape[0], -1)
        table = wandb.Table(["plot_id", "x1_orig", "x2_orig", "x1_emb", "x2_emb", "category_color", "assigned_color"])
        for i in range(self._plot_next_embeddings):
            required_colors = max(torch.max(categories), torch.max(assignments)) + 1
            if self.colors.shape[0] <= required_colors:
                self.colors = np.concatenate(
                    (self.colors, np.array((required_colors - self.colors.shape[0]) * ["#000"])))
            for p in range(x_coords.shape[1]):
                table.add_data(i, x_in[i, p, 0].item(), x_in[i, p, 1].item(), x_coords[i, p, 0].item(),
                               x_coords[i, p, 1].item(), self.colors[categories[i, p]], self.colors[assignments[i, p]])
        # plot = custom_logger.plot_table(f"{custom_logger.wandb_entity}/custom_scatter", table,
        #                                 {"x": "x2_emb", "x1_emb": "x2_emb", "fill_color": "category_color",
        #                                  "border_color": "assigned_color"},
        #                                 {"title": f"Embeddings"})
        # custom_logger.log({f"scatter": plot}, step=epoch)
        custom_logger.log({f"scatter": table}, step=epoch)
        self._plot_next_embeddings = 0


def train_test_epoch(train: bool, model: ClusterModule, optimizer, loader: DataLoader, epoch: int, mode: str,
                     dataset: ClusterDataset):
    if train:
        model.train()
        sum_loss = 0
        sum_classification_loss = 0
    correct = 0
    class_counts = torch.zeros(dataset.num_classes)
    with nullcontext() if train else torch.no_grad():
        for data in loader:
            x = data["x"].to(custom_logger.device)
            y = data["y"].to(custom_logger.device)
            batch_size = y.size(0)
            if train:
                optimizer.zero_grad()

            out, assignments = model(x, y=y, categories=data["categories"], dataset=dataset, epoch=epoch)
            if train:
                classification_loss = F.nll_loss(out, y)
                # Reactivate this part to compare to first of 3 methods
                # target = target.repeat(num_samples)
                # if probability_weights_type == "none":
                #     classification_loss = F.nll_loss(out, target)
                # else:
                #     assert not probabilities.requires_grad
                #     if probability_weights_type == "log_prob":
                #         probabilities = torch.log(probabilities)
                #     elif probability_weights_type != "prob":
                #         raise ValueError(f"Unknown probability weights type {probability_weights_type}!")
                #     classification_loss_per_sample = probabilities * F.nll_loss(out, target, reduction='none')
                #     sum_sample_probs += torch.sum(probabilities).item()
                #
                #     classification_loss = torch.mean(classification_loss_per_sample)
                loss = classification_loss

                sum_loss += batch_size * float(loss)
                sum_classification_loss += batch_size * float(classification_loss)

            pred_classes = out.argmax(dim=1)
            try:
                correct += int((pred_classes == y).sum())
                if mode == "test":
                    class_counts += torch.bincount(pred_classes.detach(), minlength=dataset.num_classes).cpu()
            except:
                print(pred_classes.shape, pred_classes)
                print(y.shape, y)
            if train:
                loss.backward()
                optimizer.step()
    dataset_len = len(loader.dataset)  # * num_samples
    additional_dict = {}
    class_counts /= dataset_len
    if mode == "train":
        additional_dict = {
            # f"{mode}_avg_sample_prob": sum_sample_probs / dataset_len,
            f"{mode}_loss": sum_loss / dataset_len,
            f"{mode}_classification_loss": sum_classification_loss / dataset_len,
            "num_clusters": model.blackbox.cluster_alg.centroids.shape[0]
        }
    else:
        additional_dict = {f"{mode}_percentage_class_{i}": class_counts[i] for i in range(dataset.num_classes)}
    custom_logger.log({f"{mode}_accuracy": correct / dataset_len, **additional_dict},
                      step=epoch)
    model.eval()  # make sure model is always in eval by default
    return correct / dataset_len


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument('--num_categories', type=int, default=4,
                        help='The number of different cluster categories.')
    parser.add_argument('--num_centroids_per_category', type=int, default=2,
                        help='The number of different centroids to use for each category.')
    parser.add_argument('--num_points', type=int, default=4,
                        help='The number of points in each sample.')
    parser.add_argument('--input_dim', type=int, default=2,
                        help='The dimensionality of the input points.')
    parser.add_argument('--num_samples', type=int, default=20,
                        help='The number of samples to take in the blackbox differentiation step.')
    parser.add_argument('--data_noise_std', type=int, default=0.01,
                        help='Standard deviation of the Gaussian noise added to the centroids to get the points.')

    # Model
    parser.add_argument('--intermediate_dim', type=int, default=2,
                        help='The dimensionality in which the clustering will be performed.')
    parser.add_argument('--pool_dim', type=int, default=8,
                        help='Dimensionality at which the final pooling will be performed')
    parser.add_argument('--diff_noise_std', type=float, default=0.1,
                        help='Standard deviation of the Gaussian noise added for blackbox differentiability.')
    parser.add_argument('--mean_shift_range', type=float, default=0.1,
                        help='Range of the mean shift clustering')
    parser.add_argument('--blackbox_transparency', type=float, default=0,
                        help='How much of the original gradients to use. 0 means only hyperplane approximations and 1 '
                             'only original gradients')
    # Training
    parser.add_argument('--lr', type=float, default=0.01,
                        help='The Adam learning rate to use.')
    parser.add_argument('--wd', type=float, default=5e-4,
                        help='The Adam weight decay to use.')
    parser.add_argument('--num_epochs', type=int, default=20000,
                        help='The number of epochs to train for.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='The batch size to use.')
    parser.add_argument('--num_train_samples', type=int, default=1000,
                        help='The number of samples in the training set.')
    parser.add_argument('--num_test_samples', type=int, default=200,
                        help='The number of samples in the test set.')

    # Logging & Misc
    parser.set_defaults(use_wandb=True)
    parser.add_argument('--no_wandb', action='store_false', dest='use_wandb',
                        help='Turns off logging to wandb')
    parser.add_argument('--wandb_name', type=str, default=None,
                        help="Name of the wandb run. Standard randomly generated wandb names if not specified.")
    parser.add_argument('--device', type=str, default="cuda",
                        help='The device to train on. Allows to use CPU or different GPUs.')
    parser.add_argument('--seed', type=int, default=1,
                        help='The seed used for pytorch.')
    parser.add_argument('--plot_freq', type=int, default=20,
                        help='Frequency with which to plot the embedding space.')

    args = custom_logger.init(parser.parse_args())
    device = torch.device(args.device)
    custom_logger.device = device

    dataset = ClusterDataset(args.num_categories, args.num_centroids_per_category, args.num_points, args.input_dim,
                             args.data_noise_std)
    torch.manual_seed(0)  # Ensure to always generate the same dataset
    np.random.seed(0)
    train_set = [dataset.sample() for _ in range(args.num_train_samples)]
    test_set = [dataset.sample() for _ in range(args.num_test_samples)]
    train_loader = DataLoader(train_set, batch_size=args.batch_size)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)

    model = ClusterModule(args.input_dim, args.intermediate_dim, args.pool_dim, dataset.num_classes,
                          MeanShiftWrapper(args.mean_shift_range), args.num_samples, args.diff_noise_std,
                          args.blackbox_transparency)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    torch.manual_seed(args.seed)
    for epoch in tqdm(range(args.num_epochs)):
        train_test_epoch(True, model, optimizer, train_loader, epoch, "train", dataset)
        if epoch % args.plot_freq == 0:
            model.plot_next_embeddings(64)
        train_test_epoch(False, model, optimizer, test_loader, epoch, "test", dataset)
