import torch
from torch_geometric.data import Data
device = torch.device('cuda')

import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, Linear

class MyGNN(torch.nn.Module):
    def __init__(self, dataset, layer_sizes=[16, 16, 16, 16]):
        super().__init__()

        layer_sizes = [dataset.num_node_features] + layer_sizes
        convs = []
        for i in range(len(layer_sizes) - 1):
            convs.append(GCNConv(layer_sizes[i], layer_sizes[i+1]))

        self.convs = torch.nn.ModuleList(convs)
        self.final = Linear(layer_sizes[-1], dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            # x = F.dropout(x, training=self.training)

        x = global_mean_pool(x, x.batch) # x.batch is used to only average about nodes in the same batch
        return F.log_softmax(self.final(x), dim=1)
        # x = F.dropout(x, training=self.training)



if __name__ == '__main__':

    pass