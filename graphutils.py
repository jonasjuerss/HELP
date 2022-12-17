from typing import Union, Tuple

import networkx as nx
import torch
import torch_geometric
from torch_geometric.data import Data


def adj_to_edge_index(adj):
    return adj.nonzero().t().contiguous()

def draw_graph(data : Data):
    g = torch_geometric.utils.to_networkx(data, to_undirected=True)
    colors = torch.sum(data.x * torch.arange(data.x.shape[1])[None, :], dim=1)
    nx.draw(g, node_color=colors, pos=nx.spring_layout(g, seed=1), with_labels=True)