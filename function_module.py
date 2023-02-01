import torch.nn


class FunctionModule(torch.nn.Module):

    def __init__(self, fun, **kwargs):
        super().__init__()
        self.fun = fun
        self.kwargs = kwargs

    def forward(self, x):
        return self.fun(x, **self.kwargs)