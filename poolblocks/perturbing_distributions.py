import abc

import torch
from torch.distributions import Normal

import custom_logger
from data_generation.serializer import ArgSerializable


class PerturbingDistribution(ArgSerializable, abc.ABC):
    
    def __init__(self, **kwargs):
        super().__init__(kwargs)

    def __call__(self, x: torch.Tensor, num_samples: int = 1):
        """

        :param x: [num_nodes, embedding_size] vectors to be perturbed
        :param num_samples: number of times to repeat all points before applying the perturbation
        :return: [num_samples * num_nodes, embedding_size]
        """
        return self._perturb(x.repeat(num_samples, 1))

    def sample(self, shape):
        raise NotImplementedError(f"The perturbation {self.__class__.__name__} does not support sampling independent of"
                                  f" input!")

    def _perturb(self, x: torch.Tensor):
        return x + self.sample(x.shape)
class GaussianPerturbation(PerturbingDistribution):

    def __init__(self, std: float):
        """
        :param std: The standard deviation sigma
        """
        super().__init__(std=std)
        self.distr = Normal(loc=torch.tensor(0.0, device=custom_logger.device),
                            scale=torch.tensor(std, device=custom_logger.device))

    def sample(self, shape):
        return self.distr.sample(shape)
