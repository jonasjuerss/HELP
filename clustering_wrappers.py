from __future__ import annotations

import abc
import math
from typing import Union, Optional, Type

import torch
from torch_scatter import scatter

import custom_logger
import graphutils
from kmeans import KMeans


class ClusterAlgWrapper(abc.ABC):

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit_predict(self, X: torch.Tensor, train: bool = False) -> torch.Tensor:
        """

        :param X: [num_points, feature_dim]
        :param centroids: [num_centroids, feature_dim]
        :return: [num_points] (integer/long tensor with values in {0, num_centroids - 1})
        """
        self.fit(X, train=train)
        return self.predict(X)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        if self.centroids is None:
            raise ValueError("predict() called before first fit()!")
        return torch.argmin(torch.cdist(X, self.centroids), dim=-1)

    @abc.abstractmethod
    def fit(self, X: torch.Tensor, train: bool = False) -> None:
        """
        Fits to the given points and returns itself for convenience
        """
        pass

    def fit_copy(self, X: torch.Tensor, train: bool = False) -> ClusterAlgWrapper:
        """
        Used to avoid problems when calling fit() in parallel.
        By default creates a copy using the kwargs given to the super constructor and fits it to the given points.
        Can be overwritten by subclasses e.g. for efficiency where necessary.
        """
        res = self.__class__(**self.kwargs)
        res.fit(X)
        return res

    @property
    @abc.abstractmethod
    def centroids(self) -> torch.Tensor:
        pass


class KMeansWrapper(ClusterAlgWrapper):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # For backward compatibility:
        kwargs_map = dict(num_concepts="n_clusters", kmeans_threshold="threshold", cluster_threshold="threshold")
        for k, v in kwargs_map.items():
            if k in kwargs:
                kwargs[v] = kwargs[k]
                del kwargs[k]
        self.kmeans = KMeans(**kwargs)

    def fit(self, X: torch.Tensor, train: bool = False) -> None:
        self.kmeans.fit(X)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return self.kmeans.predict(X)

    def fit_predict(self, X: torch.Tensor, train: bool = False) -> torch.Tensor:
        return self.kmeans.fit_predict(X)

    @property
    def centroids(self) -> torch.Tensor:
        return self.kmeans.centroids


# class MeanShiftWrapper(ClusterAlgWrapper):
#
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.meanshift = MeanShiftEuc(**kwargs)
#
#     def fit(self, X: torch.Tensor) -> None:
#         self.meanshift.fit(X)
#
#     def predict(self, X: torch.Tensor) -> torch.Tensor:
#         return self.meanshift.predict(X)
#
#     @property
#     def centroids(self) -> torch.Tensor:
#         return self.meanshift.cluster_centers_

def get_from_name(name: str) -> Type[ClusterAlgWrapper]:
    return globals()[name + "Wrapper"]


class MeanShiftWrapper(ClusterAlgWrapper):
    def __init__(self, range: float):
        super().__init__(range=range)
        self.range = range
        self._centroids = None

    def fit(self, X: torch.Tensor, train: bool = False) -> None:
        centroids = X
        mask_prev = None
        mask = None
        while mask_prev is None or not torch.equal(mask, mask_prev):
            mask_prev = mask
            # [num_centroids_new, num_centroids] boolean mask of  points in the area
            mask = torch.unique(torch.cdist(centroids, centroids) < self.range, dim=0)
            # from here on we, basically calculate mask @ centroids / sum(mask, dim=1) in a sparse/more efficient way
            # [num_points_in_mask, 2] indices of the points
            indices = torch.argwhere(mask)
            # [num_points_in_mask, feature_dim] for each point in a radius, all coordinates (note that we ignore
            # coordinate 0 which just gives) us the row of centroids
            values = centroids[indices[:, 1]]
            # sparse tensor [num_windows, num_points, feature_dim]
            sparse_tensor = torch.sparse_coo_tensor(indices.T, values, size=(mask.shape[0], mask.shape[1], X.shape[1]))
            # [num_centroids, feature_dim]
            centroids = torch.sparse.sum(sparse_tensor, dim=1).to_dense() / torch.sum(mask, dim=1)[:, None]
        self._centroids = centroids

    @property
    def centroids(self) -> torch.Tensor:
        return self._centroids

class SequentialKMeansMeanShiftWrapper(torch.nn.Module, ClusterAlgWrapper):
    """
    Idea: Inspired by https://qr.ae/pypKoP and the linked paper. Maintain overestimated number of sketches and counts
    for each of them. Update according to (https://stackoverflow.com/a/3706827) and then also calculate the actually
    used clusters from those high-level ones. Could either use my distance-metric based one again, or do whatever they
    do in the paper (might be smarter)

    For the clustering of sketches I could also use mean-shift

    Note that I'm not exactly using the same mechanism for sequential stuff sa I update with the whole batch for
    computational efficiency

    We eliminate outliers / artifact from previous batches by eliminating all meanshift clusters under some threshold
    --------------
    Old Idea: Maintain centroids and counts. On each batch, update closed centroid if under some relative threshold,
    otherwise create a new cluster
    Some limited inspiration drawn from:
    """

    def __init__(self, num_sketches: int, mean_shift_range: float, min_samples_per_sketch: float,
                 cluster_decay_factor: float = 1, rescale_clusters_decay: float = -1):
        super().__init__()
        self.num_sketches = num_sketches
        self.decay_factor = cluster_decay_factor
        self.min_samples_per_sketch = min_samples_per_sketch
        self.mean_shift_range = mean_shift_range
        self.sketches = None
        self._centroids = None
        self.counts = None
        self.rescale_clusters_decay = rescale_clusters_decay
        self.running_std = None

    def dense_mean_shift(self, X: torch.Tensor) -> torch.Tensor:
        """
        Note that here the number of points will be relatively small (< 100) so converting to sparse might induce a
        bigger overhead than just using the dense implementation
        """
        centroids = X
        mask_prev = None
        mask = None
        while mask_prev is None or not torch.equal(mask, mask_prev):
            mask_prev = mask
            # [num_centroids_new, num_centroids] boolean mask of  points in the area
            mask = torch.unique(torch.cdist(centroids, centroids) < self.mean_shift_range, dim=0)
            centroids = (mask.float() @ centroids) / torch.sum(mask, dim=1, keepdim=True)
        return centroids

    def fit(self, X: torch.Tensor, train: bool = False):
        if not train:
            return
        X = X.detach()
        if self.rescale_clusters_decay != -1:
            if self.running_std is None:
                self.running_std = (1 - self.rescale_clusters_decay) * torch.std(X, dim=0, keepdim=True)
            else:
                self.running_std = self.rescale_clusters_decay * self.running_std +\
                                   (1 - self.rescale_clusters_decay) * torch.std(X, dim=0, keepdim=True)
            X = X / self.running_std
        if self.sketches is None:
            kmeans = KMeans(n_clusters=self.num_sketches)
            closest = kmeans.fit_predict(X)
            self.sketches = kmeans.centroids
            self.counts = torch.bincount(closest).float()
        else:
            # [num_points, num_sketches]
            closest = torch.argmin(torch.cdist(X, self.sketches), dim=1)
            # [num_sketches]
            new_counts = torch.bincount(closest, minlength=self.num_sketches)
            self.counts *= self.decay_factor
            # TODO here I'm potentially adding a large (sum of all points in sketch so far) to a small (sum of all close
            #  points in this batch) number which might yield numerical instability. Hopefully, this shouldn't be too
            #  much of a problem because of the decay.
            update_mask = new_counts > 0
            self.sketches[update_mask] = scatter(X, closest, dim=0, reduce="sum", dim_size=self.num_sketches)[update_mask] +\
                                         self.counts[update_mask, None] * self.sketches[[update_mask]]
            self.sketches[update_mask] /= (new_counts[update_mask] + self.counts[update_mask])[:, None]
            self.counts += new_counts
        self._centroids = self.dense_mean_shift(self.sketches[self.counts >
                                                              self.min_samples_per_sketch * torch.sum(self.counts), :])
        # closest = self.kmeans.fit_predict(X)
        # # [num_concepts] with the number of points mapped to each cluster (>=1)
        # counts = torch.bincount(closest)
        # if self._centroids is None:
        #     all_centroids = self.kmeans.centroids
        #     self.counts = counts
        # else:
        #     # [num_clusters_current + num_clusters, embedding_dim]
        #     all_centroids = torch.cat((self._centroids, self.kmeans.centroids), dim=0)
        #     # [num_clusters_current + num_clusters]
        #     self.counts = torch.cat((self.counts, counts), dim=0)
        #
        # # [num_clusters + num_clusters_current, num_clusters + num_clusters_current]
        # centroid_dists = torch.cdist(all_centroids, all_centroids)
        # merge_mask = centroid_dists < self.threshold * torch.max(centroid_dists)
        # assignments = graphutils.dense_components(merge_mask[None, :, :],
        #                                           torch.ones(merge_mask.shape[0], dtype=torch.bool,
        #                                                      device=X.device)[None, :],
        #                                           is_directed=False).squeeze(0) - 1
        # closest = assignments[closest]
        # self.centroids = scatter(X, closest, dim=-2, reduce="mean")
        #
        # # of course this is a bit shady. Clusters under the threshold might be merged in kmeans, moving the overall
        # #  centroid too far away to later merge with current. So maybe I should just do KMeans without threshold and always apply it laters/ here
        # # [num_clusters_batch]
        # min_dist, nearest_cluster = torch.min(distances, dim=1)

    def fit_copy(self, X: torch.Tensor, train: bool = False) -> ClusterAlgWrapper:
        raise NotImplementedError()

    @property
    def centroids(self, train: bool = False) -> torch.Tensor:
        # Note: in particular, the default predict() implementation will continue working because we scale here
        return self._centroids if self.rescale_clusters_decay == -1 else self._centroids * self.running_std


class LearnableCentroidsWrapper(torch.nn.Module, ClusterAlgWrapper):
    """
    Currently unusable as no gradients are backpropagated to the centroids. Couldn't work brcause of the detach()es
    before clustering anyway?
    """

    def __init__(self, num_concepts: int, cluster_threshold: float, centroids_init_std: Optional[float] = None,
                 centroids_init_range: Optional[float] = None):
        super().__init__()
        self.num_centroids = num_concepts
        self.cluster_threshold = cluster_threshold
        self.init_std = centroids_init_std
        self.init_range = centroids_init_range
        if [centroids_init_std, centroids_init_range].count(None) != 1:
            raise ValueError("Exactly one of the cluster initialization parameters std and range has to be given!")
        self.centroids_dirty = True
        self._centroids = None

    def fit(self, X: torch.Tensor, train: bool = False) -> None:
        if self._centroids is None:
            if self.init_range is None:
                init = self.init_std * torch.randn(self.num_centroids, X.shape[1], device=custom_logger.device)
            else:
                init = (torch.rand(self.num_centroids, X.shape[1], device=custom_logger.device) - 0.5) *\
                       (2 * self.init_range)
            self._centroids = torch.nn.Parameter(init)

            if self.cluster_threshold != 0:
                def hook_fn(x):
                    self.centroids_dirty = True
                    return x
                self._centroids.register_hook(hook_fn)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return torch.argmin(torch.cdist(X, self.centroids), dim=-1)

    def fit_copy(self, X: torch.Tensor, train: bool = False) -> ClusterAlgWrapper:
        raise NotImplementedError()

    @ClusterAlgWrapper.centroids.getter
    def centroids(self) -> torch.Tensor:
        if self.cluster_threshold == 0:
            return self._centroids

        if self.centroids_dirty:
            centroid_dists = torch.cdist(self._centroids, self._centroids)
            merge_mask = centroid_dists < self.cluster_threshold * torch.max(centroid_dists)
            # Note: there might be chains of centroids a-b-c, where dist(a, b), dist(b, c) < threshold,
            # but dist(a, c) > threshold. We decide to merge those by performing a connected component search on a graph
            # where there is an edge between 2 clusters iff. they are closer than the threshold.
            # [num_clusters] with values in [0, num_merged_clusters - 1]
            assignments = graphutils.dense_components(merge_mask[None, :, :],
                                                      torch.ones(self.num_centroids, dtype=torch.bool,
                                                                 device=custom_logger.device)[None, :],
                                                      is_directed=False).squeeze(0) - 1
            self.merged_centroids = scatter(self._centroids, assignments, dim=-2, reduce="mean")
            self.centroids_dirty = False
        return self.merged_centroids