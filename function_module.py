from typing import List

import torch.nn


class FunctionModule(torch.nn.Module):

    def __init__(self, fun, arg_names: dict, **kwargs):
        """

        :param fun: The function to call
        :param arg_names: to which parameter of fun each input argument should be mapped. E.g. {"a": "c", "b": "d"}
         would call f(a=kwargs[c], b=kwargs[d])
        :param kwargs: Additional keyword args to pass to each function call
        """
        super().__init__()
        self.fun = fun
        self.arg_names = arg_names
        self.kwargs = kwargs

    def forward(self, **kwargs):
        return self.fun(**{k: kwargs[v] for k, v in self.arg_names.items()}, **self.kwargs)

class MaskedFlatten(torch.nn.Module):
    def forward(self, input, batch_or_mask):
        """
        :param x: [batch_size, max_num_output_nodes, num_features]
        :param mask: [batch_size, max_num_output_nodes] (booleans)
        :return: [batch_size, num_output_nodes * num_features]
        """
        return input[batch_or_mask].reshape(*input.shape[:-2], -1)

class MaskedMean(torch.nn.Module):
    def forward(self, input, batch_or_mask):
        """
        :param x: [batch_size, max_num_output_nodes, num_features]
        :param mask: [batch_size, max_num_output_nodes] (booleans)
        :return: [batch_size, num_output_nodes * num_features]
        """
        # If graphs with 0 nodes were valid we could use: torch.clamp(torch.sum(mask, dim=-1), min=1)
        return torch.sum(input * batch_or_mask[..., None], dim=-2) / torch.sum(batch_or_mask, dim=-1)[..., None]

class MaskedSum(torch.nn.Module):
    def forward(self, input, batch_or_mask):
        """
        :param x: [batch_size, max_num_output_nodes, num_features]
        :param mask: [batch_size, max_num_output_nodes] (booleans)
        :return: [batch_size, num_output_nodes * num_features]
        """
        return torch.sum(input * batch_or_mask[..., None], dim=-2)

# class MaskedFunctionModule(FunctionModule):
#
#     def __init__(self, fun, arg_names: List[str], **kwargs):
#         super().__init__(fun, arg_names, **kwargs)
#
#     def forward(self, x, mask, *args, **kwargs):
#         args = (x[mask], ) + args
#         return self.fun({self.arg_names[i]: args[i] for i in range(len(args))}, **kwargs, **self.kwargs)
