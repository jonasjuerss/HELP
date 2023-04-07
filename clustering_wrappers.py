import abc
import math
from typing import Union, Optional, Type

import torch
from fast_pytorch_kmeans import KMeans


class ClusterAlgWrapper(abc.ABC):

    def __init__(self, **kwargs):
        pass
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
        pass

    @property
    @abc.abstractmethod
    def centroids(self) -> torch.Tensor:
        pass

class KMeansWrapper(ClusterAlgWrapper):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # For backward compatibility:
        if "num_concepts" in kwargs:
            kwargs["n_clusters"] = kwargs["num_concepts"]
            del kwargs["num_concepts"]
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
    def __init__(self, range: int):
        super().__init__()
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
            centroids = torch.sparse.sum(sparse_tensor, dim=1).to_dense() / torch.sum(mask, dim=1)
        self._centroids = centroids

    @property
    def centroids(self) -> torch.Tensor:
        return self._centroids
