import math
from typing import Tuple, Optional

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.utils import to_scipy_sparse_matrix, to_dense_adj, to_dense_batch
from torch_scatter import scatter

import custom_logger


def adj_to_edge_index(adj: torch.Tensor, mask: Optional[torch.Tensor] = None)\
        -> Tuple[torch.Tensor, Optional[torch.Tensor], int]:
    """

    :param adj: [max_num_nodes, max_num_nodes] or [batch_size, max_num_nodes, max_num_nodes]
    :param mask: None or [max_num_nodes] or [batch_size, max_num_nodes]
    :return:
        edge_index: [2, num_edges]
        batch: in case of back dimension: [num_nodes_total]
        num_nodes: total number of nodes
    """
    if adj.ndim == 2:
        if mask is not None:
            # There should be an easier wys to index values at the mask
            adj = adj[torch.logical_and(mask[0][None, :], mask[0][:, None])]
            num_nodes = math.isqrt(adj.shape[0])
            adj = adj.view(num_nodes, num_nodes)
        return adj.nonzero().t().contiguous(), None, adj.shape[-1]
    elif adj.ndim == 3:
        masks = torch.logical_and(mask[:, None, :], mask[:, :, None])
        num_nodes = 0
        edge_index = torch.empty(2, 0, device=custom_logger.device)
        batch = torch.empty(0, device=custom_logger.device, dtype=torch.long)
        for i in range(adj.shape[0]):
            cur_adj = adj[i][masks[i, :, :]]
            cur_nodes = math.isqrt(cur_adj.shape[0])
            cur_adj = cur_adj.view(cur_nodes, cur_nodes)
            edge_index = torch.cat((edge_index, cur_adj.nonzero().t().contiguous() + num_nodes), dim=1)
            batch = torch.cat((batch, i * torch.ones(cur_nodes, device=custom_logger.device, dtype=torch.long)), dim=0)
            num_nodes += cur_nodes
        return edge_index, batch, num_nodes
    else:
        raise ValueError(f"Unsupported number of dimensions: {adj.ndim}. The only supported formats are "
                         f"[num_nodes, num_nodes] and [batch_size, max_num_nodes, max_num_nodes]!")

def draw_graph(data : Data):
    g = torch_geometric.utils.to_networkx(data, to_undirected=True)
    colors = torch.sum(data.x * torch.arange(data.x.shape[1])[None, :], dim=1)
    nx.draw(g, node_color=colors, pos=nx.spring_layout(g, seed=1), with_labels=True)

def sparse_components(edge_index: torch.Tensor, num_nodes: int, connection="strong") -> Tuple[int, torch.Tensor]:
    """
    :param edge_index:  [2, num_edges]
    :param num_nodes:
    :return:
        num_components
        component: [num_nodes] integers mapping each node to a component
    """
    adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)
    # component: [num_nodes]
    num_components, component = sp.csgraph.connected_components(adj, connection=connection)
    return num_components, torch.tensor(component, device=custom_logger.device)

def dense_components(adj: torch.Tensor, mask: Optional[torch.Tensor], connection="strong"):
    """

    :param adj: [max_num_nodes, max_num_nodes] or [batch_size, max_num_nodes, max_num_nodes]
    :param mask: None or [max_num_nodes] or [batch_size, max_num_nodes]
    :return:
        max_num_components: maximum number of components in any graph
        component: [batch_size, max_num_nodes] integers mapping each node to a component. Starting
    """
    edge_index, batch, num_nodes = adj_to_edge_index(adj, mask)
    # TODO check if iterating over all samples and therefore having significantly less nodes per search is more efficient despite the overhead of more conversions
    # component: [num_nodes_total]
    num_components, component = sparse_components(edge_index, num_nodes, connection)
    # [batch_size] minimum component index for each batch element
    component_starts = scatter(component, batch, reduce="min")
    # [batch_size, max_num_components] (where max_num_components is the maximum number of components in any single batch element)
    dense_component, mask_new = to_dense_batch(component, batch, max_num_nodes=adj.shape[-1])
    # Subtract start component for each batch element and add 1 as 0 is a dummy concept for masked nodes and the actual
    # ones start at 1.
    dense_component = dense_component - component_starts[:, None] + 1
    # Whereas we initialized the masked components as -1, they became smaller in later batch elements as we subtracted
    # the start component. So we fix this here.
    dense_component = torch.maximum(dense_component, torch.tensor([0], device=custom_logger.device))
    return dense_component.shape[1], dense_component

def batch_from_mask(mask: torch.Tensor, max_num_nodes: int):
    # Arange one number for each batch entry, repeat them to a [batch_size, max_num_nodes] array and apply the mask
    return torch.arange(mask.shape[0], device=mask.device)[:, None].repeat_interleave(max_num_nodes, dim=1)[mask]
