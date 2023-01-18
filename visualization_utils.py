import torch
from torch_geometric.data import Data

from custom_net import CustomNet


def draw_diffpool_assignments(model: CustomNet, example : Data,
                              cluster_colors=[[1., 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [0, 0, 0]]):
    cluster_colors = torch.tensor(cluster_colors)[None, :, :]
    out, _, pool_assignments = model(example.to(device)) # Do I need to add a batch dimension to the example?
    for assignment in pool_assignments[:1]:
        assignment = torch.softmax(assignment, dim=-1) # usually performed by diffpool function
        assignment = assignment.detach().cpu().squeeze(0) # remove batch dimensions
        if cluster_colors.shape[1] < assignment.shape[1]:
            raise ValueError(f"Only {cluster_colors.shape[1]} colors given to distinguish {assignment.shape[1]} cluster")
        example.edge_index = adj_to_edge_index(example.adj)
        g = torch_geometric.utils.to_networkx(example, to_undirected=True)
        # intermediate dimensions: num_nodes x num_clusters x 3
        colors = torch.sum(assignment[:, :, None] * cluster_colors[:, :assignment.shape[1], :], dim=1)
        nx.draw(g, node_color=colors, pos=nx.spring_layout(g, seed=1), with_labels=True)