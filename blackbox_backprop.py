import abc
import math
from typing import Callable, Tuple, Any

import torch.nn
from torch.autograd.function import once_differentiable


# TODO if this doesn't perform well, definitely look at "Locally fitting hyperplanes to high-dimensional data" again

# TODO definitely try this by approximating the super simple function from the notebook and visualizing the autograd
#  gradients as a sanity check
# TODO add an additional, decaying weight such that in the end the smoothened (and thus wrong) gradients become less important and we can converge to the exact result

# TODO it should be possible to derive an error bound as this is basically taylor expansion. Obviously, we need to take
#  into account the random sampling etc.

# TODO possible improvement: if we are already calculating the forward pass for the Monte Carlo samples, we could also
#  use this to accumulated additional gradients for the part after the blackbox function. This would not come at 0 cost
#  as we would now additionally calculated the backward passes but of course it's kind of "discounted" as we already
#  have the forward pass. However, it might also not be helpful as the part trained with exact gradients might converge
#  faster anyway, so it would be counterproductive to dedicate a bigger part of the training time to it
class BlackBoxFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, num_samples: int, noise_distr, non_batch_dims: int,
                hard_fn: Callable[..., Tuple], **kwargs) -> Tuple:
        """ Runs the hard function for forward, cache the output and returns.
        All hard functions should inherit from this, it implements the autograd override.
        :param ctx: pytorch context, automatically passed in.
        :param x: input tensor.
        :param hard_fn: the non-differentiable function. Returns the output w.r.t. to which we want to differentiate
        followed by an arbitrary number- of other outputs (e.g. for logging). For efficiency, this is expected to always
        return a tuple, even if it only consists of one element
        :param kwargs: list of args to pass to hard function. Currently does not support tensors
        :returns: hard_fn(tensor), backward pass using DAB.
        """
        all_outputs = hard_fn(x, **kwargs)
        # saveable_args = list([a for a in kwargs if isinstance(a, torch.Tensor)])
        ctx.save_for_backward(x, all_outputs[0])# , *saveable_args)
        ctx.hard_fn = hard_fn
        ctx.kwargs = kwargs
        ctx.num_samples = num_samples
        ctx.non_batch_dims = non_batch_dims
        ctx.noise_distr = noise_distr # we need to ensure parameters are on the correct device
        return all_outputs

    @staticmethod
    @once_differentiable # Might also be twice but I'm not sure
    def backward(ctx, grad_out):
        """

        In the simplified case of one input/output dimension, for each batch entry, the value i of grad_out is:

        .. math::
            \frac{\partial L}{\partial out_i}

        where L denotes the overall loss/final scalar we backpropagate from and out_i denotes output i of this function.
        We calculate

        .. math::
            \frac{\partial L}{\partial in_j}


        :param ctx: pytorch context, automatically passed in.
        :param grad_out: [batch_dims, output_dims]
        :returns: [batch_dims, input_dims]
        """
        if not ctx.needs_input_grad[0]:
            return None, None, None, None, None

        # x: [batch_dims, input_dims], out: [batch_dims, output_dims]
        x, out = ctx.saved_tensors
        target_dim = x.ndim - ctx.non_batch_dims
        total_batch_dim = math.prod(x.shape[:target_dim])

        # [batch_dims, num_samples, input_dims]
        noise = ctx.noise_distr.sample(x.shape[:-ctx.non_batch_dims] + (ctx.num_samples, ) + x.shape[-ctx.non_batch_dims:])
        # [batch_dims, num_samples, input_dims]
        x_perturbed = x.unsqueeze(-ctx.non_batch_dims - 1) + noise
        #  (flatten and reshape in case hard_fn doesn't support multiple batch dimensions)
        out_perturbed = ctx.hard_fn(x_perturbed.flatten(target_dim - 1, target_dim), **ctx.kwargs)[0]
        # [batch_dims, num_samples, output_dims]
        # print(out.shape, out.unsqueeze(target_dim).shape, out_perturbed.shape, x_perturbed.shape, x_perturbed.flatten(target_dim - 1, target_dim).shape, target_dim, noise.shape)
        out_perturbed.reshape(*out_perturbed.shape[:target_dim - 1], -1, ctx.num_samples, *out_perturbed.shape[target_dim:])
        # [batch_dims, num_samples + 1, output_dims]

        out_perturbed = torch.cat((out.unsqueeze(target_dim), out_perturbed), dim=target_dim)
        # [batch_dims, num_samples + 1, total_in_dims]
        x_perturbed = torch.cat((x.unsqueeze(target_dim), x_perturbed), dim=target_dim).flatten(target_dim + 1)
        # [batch_dims, num_samples + 1, input_dims + 1] added constant 1 entries to input for bias
        x_perturbed = torch.cat((x_perturbed, torch.ones(*x_perturbed.shape[:-1], 1, device=x.device)), dim=-1)
        # [batch_dims, total_in_dims + 1, total_out_dims]
        approx_factors = torch.linalg.lstsq(x_perturbed, out_perturbed.flatten(target_dim).reshape(*out_perturbed.shape[:target_dim + 1], -1)).solution
        # [batch_dims, total_in_dims, total_out_dims] for each batch entry and each input i: partial L / partial
        approx_factors = approx_factors[..., :-1, :]
        # [total_batch_dim, 1, total_out_dim]
        grad_prod = grad_out.reshape(total_batch_dim, 1, -1).bmm(approx_factors.reshape(total_batch_dim, *approx_factors.shape[-2:]).transpose(1, 2))
        return grad_prod, None, None, None, None


class BlackBoxSkipFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, res: torch.Tensor, num_samples: int, noise_distr, non_batch_dims: int,
                hard_fn: Callable[..., Tuple], **kwargs) -> torch.Tensor:
        ctx.save_for_backward(x, res)
        ctx.hard_fn = hard_fn
        ctx.kwargs = kwargs
        ctx.num_samples = num_samples
        ctx.non_batch_dims = non_batch_dims
        ctx.noise_distr = noise_distr
        return res

    @staticmethod
    @once_differentiable # Might also be twice but I'm not sure
    def backward(ctx, grad_out):
        """

        In the simplified case of one input/output dimension, for each batch entry, the value i of grad_out is:

        .. math::
            {\partial L}/{\partial out_i}

        where L denotes the overall loss/final scalar we backpropagate from and out_i denotes output i of this function.
        We calculate

        .. math::
            {\partial L}/{\partial in_j}

        :param ctx: pytorch context, automatically passed in.
        :param grad_out: [batch_dims, output_dims]
        :returns: [batch_dims, input_dims]
        """
        if not ctx.needs_input_grad[0]:
            return (None, ) * (6 + len(ctx.kwargs))
        # x: [batch_dims, input_dims], out: [batch_dims, output_dims]
        x, out = ctx.saved_tensors
        target_dim = x.ndim - ctx.non_batch_dims
        total_batch_dim = math.prod(x.shape[:target_dim])
        #print(f"x: {x.shape}, out: {out.shape}, target_dim: {target_dim}, total_batch_dim: {total_batch_dim}")

        # [batch_dims, num_samples, input_dims]
        noise = ctx.noise_distr.sample(x.shape[:-ctx.non_batch_dims] + (ctx.num_samples, ) + x.shape[-ctx.non_batch_dims:])
        # [batch_dims, num_samples, input_dims]
        x_perturbed = x.unsqueeze(-ctx.non_batch_dims - 1) + noise
        #  (flatten and reshape in case hard_fn doesn't support multiple batch dimensions)
        out_perturbed = ctx.hard_fn(x_perturbed.flatten(target_dim - 1, target_dim), **ctx.kwargs)[0]
        # [batch_dims, num_samples, output_dims]
        #print(f"out_perturbed: {out_perturbed.shape}, x_perturbed: {x_perturbed.shape}, noise: {noise.shape}")
        out_perturbed = out_perturbed.reshape(*out_perturbed.shape[:target_dim - 1], -1, ctx.num_samples, *out_perturbed.shape[target_dim:])
        #print(f"out_perturbed: {out_perturbed.shape}")
        # [batch_dims, num_samples + 1, output_dims]
        out_perturbed = torch.cat((out.unsqueeze(target_dim), out_perturbed), dim=target_dim)
        # [batch_dims, num_samples + 1, total_in_dims]
        x_perturbed = torch.cat((x.unsqueeze(target_dim), x_perturbed), dim=target_dim).flatten(target_dim + 1)
        # [batch_dims, num_samples + 1, input_dims + 1] added constant 1 entries to input for bias
        x_perturbed = torch.cat((x_perturbed, torch.ones(*x_perturbed.shape[:-1], 1, device=x.device)), dim=-1)
        # [batch_dims, total_in_dims + 1, total_out_dims]
        approx_factors = torch.linalg.lstsq(x_perturbed, out_perturbed.flatten(target_dim).reshape(*out_perturbed.shape[:target_dim + 1], -1)).solution
        # [batch_dims, total_in_dims, total_out_dims] for each batch entry and each input i: partial L / partial
        approx_factors = approx_factors[..., :-1, :]
        # [total_batch_dim, 1, total_in_dim]
        grad_prod = grad_out.reshape(total_batch_dim, 1, -1).bmm(
            approx_factors.reshape(total_batch_dim, *approx_factors.shape[-2:]).transpose(1, 2))
        #print(grad_prod.shape)
        grad_prod = grad_prod.reshape(ctx.saved_tensors[0].shape)
        return grad_prod, *((None, ) * (5 + len(ctx.kwargs)))

class BlackBoxModule(torch.nn.Module, abc.ABC):

    def __init__(self, num_samples: int, noise_distr: torch.distributions.Distribution, non_batch_dims: int):
        """

        :param num_samples: Number of monte carlo samples to take for the hyperplane approximation
        :param noise_distr: distribution to sample noise from
        :param non_batch_dims: number of dimensions that are associated with a single sample. The rest will be
            interpreted as batch dimensions
        """
        super().__init__()
        self.num_samples = num_samples
        self.noise_distr = noise_distr
        self.non_batch_dims = non_batch_dims
        self._full_fun = lambda x, **kwargs: self.postprocess(*self.hard_fn(x, **kwargs))

    @abc.abstractmethod
    def hard_fn(self, x: torch.Tensor, **kwargs) -> Tuple:
        """
        The non-differentiable function
        :param x: The first argument to the non-differentiable. This needs to be the (only) one that the rest of the
            network backpropagates with respect to
        :param kwargs: Other input the  blackbox function through which we cannot backpropagate
        :return: A Tuple of results of this blackbox function. The first one needs to be the only tensor we
        backpropagate w.r.t
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def postprocess(self, x: torch.Tensor, *args) -> Tuple:
        """
        Postprocesses
        :param x: The first entry returned by hard_fn. This needs to be the (only) one that the rest of the network
            backpropagates with respect to
        :param args: The remaining output of hard_fn()
        :return: A Tuple of results for this module. The first one needs to be the only one that the rest of the networ
            backpropagates w.r.t.
        """
        raise NotImplementedError()

    def forward(self, x: torch.Tensor, **kwargs):
        # We compute the hard_fn and postprocess step outside of the function to allow gradient flow into postprocess.
        # However, we detach the input of postprocess and instead return a result from our blackbox function which just
        # returns what we already calculated but will return the approximated gradients for backpropagation before this
        # module
        res_tuple = self.hard_fn(x, **kwargs)
        final_res = self.postprocess(res_tuple[0].detach(), *res_tuple[1:])
        return BlackBoxSkipFun.apply(x, final_res[0], self.num_samples, self.noise_distr, self.non_batch_dims,
                                     self._full_fun, **kwargs), *final_res[1:]


# class BaseHardFn(torch.autograd.Function):
# """
# Taken from Ramapuram et. al. (2019) https://arxiv.org/pdf/1905.03658.pdf
# """
#     @staticmethod
#     def forward(ctx, x, soft_y, hard_fn, *args):
#         """ Runs the hard function for forward, cache the output and returns.
#         All hard functions should inherit from this, it implements the autograd override.
#         :param ctx: pytorch context, automatically passed in.
#         :param x: input tensor.
#         :param soft_y: forward pass output (logits) of DAB approximator network.
#         :param hard_fn: to be passed in from derived class.
#         :param args: list of args to pass to hard function.
#         :returns: hard_fn(tensor), backward pass using DAB.
#         :rtype: torch.Tensor
#         """
#         hard = hard_fn(x, *args)
#         saveable_args = list([a for a in args if isinstance(a, torch.Tensor)])
#         ctx.save_for_backward(x, soft_y, *saveable_args)
#         return hard
#     @staticmethod
#     def _hard_fn(x, *args):
#         raise NotImplementedError("implement _hard_fn in derived class")
#     @staticmethod
#     def backward(ctx, grad_out):
#         """ Returns DAB derivative.
#         :param ctx: pytorch context, automatically passed in.
#         :param grad_out: grads coming into layer
#         :returns: dab_grad(tensor)
#         :rtype: torch.Tensor
#         """
#         x, soft_y, *args = ctx.saved_tensors
#         with torch.enable_grad():
#             grad = torch.autograd.grad(outputs=soft_y, inputs=x, grad_outputs=grad_out, retain_graph=True)
#         return grad[0], None, None, None
# class SignumWithMargin(BaseHardFn):
#     @staticmethod
#     def _hard_fn(x, *args):
#         """ x[x < -eps] = -1
#         x[x > +eps] = 1
#         else x = 0
#         :param x: input tensor
#         :param args: list of args with 0th element being eps
#         :returns: signum(tensor)
#         :rtype: torch.Tensor
#         """
#         eps = args[0] if len(args) > 0 else 0.5
#         sig = torch.zeros_like(x)
#         sig[x < -eps] = -1
#         sig[x > eps] = 1
#         return sig
#     @staticmethod
#     def forward(ctx, x, soft_y, *args):
#         return BaseHardFn.forward(ctx, x, soft_y, SignumWithMargin._hard_fn, *args)