import abc
import math
from typing import Union, Optional, Type

import torch
from fast_pytorch_kmeans import KMeans

from meanshift.mean_shift_gpu import MeanShiftEuc


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

# class MeanShift(CustomClusteringAlgorithm):
#     """
#     Inspired by: https://colab.research.google.com/github/sotte/pytorch_tutorial/blob/master/notebooks/mean_shift_clustering.ipynb
#     """
#
#     def __init__(self, bandwidth: int, batch_size: int = 1000):
#         super().__init__()
#         self.bandwidth = bandwidth  # TODO this seems to be a super critical parameter, choose wisely
#         self.batch_size = batch_size
#         self.divisor = self.bandwidth * math.sqrt(2 * math.pi)
#
#
#     def fit_predict(self, X: torch.Tensor, centroids: torch.Tensor = None) -> torch.Tensor:
#         for _ in range(self.num_iterations):
#             for i in range(0, X.shape[0], self.batch_size):
#                 end_index = min(X.shape[0], i + self.batch_size)
#                 # [num_points, (remainder of) batch_size] pairwise distance between each point and each point in the batch
#                 dist = torch.cdist(X, X[i:end_index])
#                 # [num_points, (remainder of) batch_size]
#                 weight = torch.exp(-0.5 * (dist / self.bandwidth) ** 2) / self.divisor
#                 num = (weight[:, :, None] * X[None, :, :]).sum(dim=1)
#                 X[i:end_index] = num / weight.sum(dim=1)[:, None]
#         return X
#
#     def fit(self, X: torch.Tensor, centroids: Optional[torch.Tensor] = None) -> None:
#         pass
#
#     def predict(self, X: torch.Tensor) -> torch.Tensor:
#         pass
