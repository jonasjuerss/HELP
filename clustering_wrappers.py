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

    def fit_predict(self, X: torch.Tensor) -> torch.Tensor:
        """

        :param X: [num_points, feature_dim]
        :param centroids: [num_centroids, feature_dim]
        :return: [num_points] (integer/long tensor with values in {0, num_centroids - 1})
        """
        self.fit(X)
        return self.predict(X)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        if self.centroids is None:
            raise ValueError("predict() called before first fit()!")
        return torch.argmin(torch.cdist(X, self.centroids), dim=-1)

    @abc.abstractmethod
    def fit(self, X: torch.Tensor) -> None:
        """
        Fits to the given points and returns itself for convenience
        """
        pass

    def fit_copy(self, X: torch.Tensor) -> ClusterAlgWrapper:
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

    def fit(self, X: torch.Tensor) -> None:
        self.kmeans.fit(X)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return self.kmeans.predict(X)

    def fit_predict(self, X: torch.Tensor) -> torch.Tensor:
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


    def fit(self, X: torch.Tensor) -> None:
        centroids = X
        mask_prev = None
        mask = None
        while mask_prev is None or not torch.equal(mask, mask_prev):
            mask_prev = mask
            # [num_centroids, num_points] boolean mask of  points in the area
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


class LearnableCentroidsWrapper(torch.nn.Module, ClusterAlgWrapper):

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

    def fit(self, X: torch.Tensor) -> None:
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

    def fit_copy(self, X: torch.Tensor) -> ClusterAlgWrapper:
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