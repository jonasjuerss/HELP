import argparse
import io
import json
import typing
from contextlib import nullcontext

import networkx as nx
import torch
import torch_geometric
from PIL import Image
from matplotlib import pyplot as plt
from torch_geometric.data import Data

import wandb
from torch_geometric import transforms
from torch_geometric.loader import DataLoader, DenseDataLoader
from torch_geometric.nn import DenseGCNConv
from tqdm import tqdm

import custom_logger
from data_generation.custom_dataset import UniqueMotifCategorizationDataset, CustomDataset
from custom_net import CustomNet
from custom_logger import log
import torch.nn.functional as F

from data_generation.deserializer import from_dict
from data_generation.motifs import BinaryTreeMotif, HouseMotif, FullyConnectedMotif
from graphutils import adj_to_edge_index

CONV_TYPES = [DenseGCNConv]
device = None

def train_test_epoch(train: bool, optimizer, loader: DenseDataLoader, epoch: int, pooling_loss_weight: float,
                     entropy_loss_weight: float):
    if train:
        model.train()
    else:
        model.eval()
    correct = 0
    sum_loss = 0
    sum_classification_loss = 0
    sum_pooling_loss = 0
    sum_entropy_loss = 0
    with nullcontext() if train else torch.no_grad():
        for data in loader:
            data = data.to(device)
            if train:
                optimizer.zero_grad()

            out, pooling_loss, _ = model(data)
            classification_loss = F.nll_loss(out, data.y.squeeze(1))
            entropy_loss = model.entropy_loss()
            loss = classification_loss + pooling_loss_weight * pooling_loss + entropy_loss_weight * entropy_loss

            batch_size = data.y.size(0)
            sum_loss += batch_size * float(loss)
            sum_classification_loss += batch_size * float(classification_loss)
            sum_pooling_loss += batch_size * float(pooling_loss)
            sum_entropy_loss += batch_size * float(entropy_loss)
            correct += int((out.argmax(dim=1) == data.y.squeeze(1)).sum())

            if train:
                loss.backward()
                optimizer.step()
    dataset_len = len(loader.dataset)
    mode = "train" if train else "test"
    log({f"{mode}_loss": sum_loss / dataset_len,
         f"{mode}_pooling_loss": sum_pooling_loss / dataset_len,
         f"{mode}_entropy_loss": sum_entropy_loss / dataset_len,
         f"{mode}_classification_loss": sum_classification_loss / dataset_len,
         f"{mode}_accuracy": correct / dataset_len},
        step=epoch,
        commit=not train)

def log_graphs(graphs: typing.List[Data],
               epoch: int,
               cluster_colors=torch.tensor([[1., 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [0, 0, 0]])[None, :, :]):
    # TODO log to tables in wandb instead of just figures. Add hover labels to see exact cluster assignment
    # node_table = wandb.Table(["graph", "pool_step", "index", "r", "g", "b", "label"])
    # edge_table = wandb.Table(["graph", "pool_step", "from", "to"])

    graphs_table = wandb.Table(columns=["graph", "pool_step", "class", "train_sample", "image"])
    with torch.no_grad():
        for graph_i, data in enumerate(graphs):
            data = data.clone().detach().to(device)
            out, _, pool_assignments = model(data)
            for pool_step, assignment in enumerate(pool_assignments[:1]):
                assignment = torch.softmax(assignment, dim=-1)  # usually performed by diffpool function
                assignment = assignment.detach().cpu().squeeze(0)  # remove batch dimensions

                if cluster_colors.shape[1] < assignment.shape[1]:
                    raise ValueError(
                        f"Only {cluster_colors.shape[1]} colors given to distinguish {assignment.shape[1]} cluster")

                # [num_nodes, 3] (intermediate dimensions: num_nodes x num_clusters x 3)
                colors = torch.sum(assignment[:, :, None] * cluster_colors[:, :assignment.shape[1], :], dim=1)[:data.num_nodes, :]
                # for i in range(colors.shape[0]):
                #     node_table.add_data(graph_i, pool_step, i, colors[i, 0].item(), colors[i, 1].item(), colors[i, 2].item())

                # [3, num_edges] where the first row seems to be constant 0, indicating the graph membership
                data.edge_index = adj_to_edge_index(data.adj)
                # for i in range(edge_index.shape[1]):
                #     edge_table.add_data(graph_i, pool_step, edge_index[1, i].item(), edge_index[2, i].item())

                # Legacy:
                im_io = io.BytesIO()
                fig, ax = plt.subplots()
                g = torch_geometric.utils.to_networkx(data, to_undirected=True)
                nx.draw(g, node_color=colors, with_labels=True, ax=ax)#, pos=nx.spring_layout(g, seed=1))
                fig.savefig(im_io, format='png')
                plt.close(fig)
                im = Image.open(im_io)
                graphs_table.add_data(graph_i, pool_step, data.y.item(), graph_i < len(graphs) / 2, wandb.Image(im))

    log(dict(
        graphs_table=graphs_table
        # node_table=node_table,
        # edge_table=edge_table
    ), step=epoch)



num_colors = 2
current_dataset = UniqueMotifCategorizationDataset(BinaryTreeMotif(4, [0], num_colors),
                                                   [HouseMotif([1], [1], num_colors),
                                                    FullyConnectedMotif(5, [1], num_colors)],
                                                   [0.6, 0.6])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.01,
                        help='The Adam learning rate to use.')
    parser.add_argument('--pooling_loss_weight', type=float, default=0.0,
                        help='The weight of the pooling loss.')
    parser.add_argument('--entropy_loss_weight', type=float, default=0.0001,
                        help='The weight of the entropy loss in the explanation layer.')
    parser.add_argument('--wd', type=float, default=5e-4,
                        help='The Adam weight decay to use.')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='The number of epochs to train for.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='The batch size to use.')
    parser.add_argument('--add_layer', type=int, nargs='+', action='append',
                        default=[[16, 16, 16, 16, 16], [16, 16, 16]], dest='layer_sizes',
                        help='The layer sizes to use. Example: --add_layer 16 32 --add_layer 32 64 16 results in a '
                             'network with 2 pooling steps where 5 message passes are performed before the first and ')
    parser.add_argument('--nodes_per_layer', type=int, default=[5, 1],
                        help='The number of nodes after each pooling step for architectures like DiffPool which require'
                             ' to pre-specify that. Note that the last one should be 1 for classification')

    parser.add_argument('--conv_type', type=str, default="DenseGCNConv",
                        help='The type of graph convolution to use.')
    parser.add_argument('--pooling_config', type=json.loads, default="{}",
                        help="A json with further arguments for the pooling type used")

    parser.add_argument('--dataset', type=json.loads, default=current_dataset.__dict__(),
                        help="A json that defines the current dataset")

    parser.add_argument('--train_set_size', type=int, default=512,
                        help='Number of samples for the training set.')
    parser.add_argument('--test_set_size', type=int, default=128,
                        help='Number of samples for the training set.')

    parser.add_argument('--graph_log_freq', type=int, default=10,
                        help='Every how many epochs to log graphs to wandb.')
    parser.add_argument('--graphs_to_log', type=int, default=3,
                        help='How many graphs from the training and testing set to log.')

    parser.set_defaults(dense_data=True)
    parser.add_argument('--spase_data', action='store_false', dest='dense_data',
                        help='Switches from a dense representation of graphs (dummy nodes are added so that each of '
                             'them has the same number of nodes) to a sparse one.')

    parser.add_argument('--seed', type=int, default=1,
                        help='The seed used for pytorch. This also determines the dataset if generated randomly.')

    parser.add_argument('--device', type=str, default="cuda",
                        help='The device to train on. Allows to use CPU or different GPUs.')
    parser.set_defaults(use_wandb=True)
    parser.add_argument('--no_wandb', action='store_false', dest='use_wandb',
                        help='Turns off logging to wandb')
    args = custom_logger.init(parser.parse_args())

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    dataset = typing.cast(CustomDataset, from_dict(args.dataset))


    conv_type = next((x for x in CONV_TYPES if x.__name__ == args.conv_type), None)
    if conv_type is None:
        raise ValueError(f"There is no convolution type named {args.conv_type}!")
    model = CustomNet(dataset.num_node_features, dataset.num_classes, layer_sizes=args.layer_sizes,
                      num_nodes_per_layer=args.nodes_per_layer, use_entropy_layer=False, conv_type=conv_type).to(device)

    train_data = [dataset.sample(dense=args.dense_data) for _ in range(args.train_set_size)]
    test_data = [dataset.sample(dense=args.dense_data) for _ in range(args.test_set_size)]
    graphs_to_log = train_data[:args.graphs_to_log] + test_data[:args.graphs_to_log]
    if args.dense_data:
        train_loader = DenseDataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        test_loader = DenseDataLoader(test_data, batch_size=args.batch_size, shuffle=True)
        log_graph_loader = DenseDataLoader(graphs_to_log, batch_size=1, shuffle=False)
    else:
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)
        log_graph_loader = DataLoader(graphs_to_log, batch_size=1, shuffle=False)
        raise NotImplementedError("Non-dense data is easy to implement if necessary")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    for epoch in tqdm(range(args.num_epochs)):
        train_test_epoch(True, optimizer, train_loader, epoch, args.pooling_loss_weight, args.entropy_loss_weight)
        if epoch % args.graph_log_freq == 0:
            log_graphs(graphs_to_log, epoch)
        train_test_epoch(False, optimizer, test_loader, epoch, args.pooling_loss_weight, args.entropy_loss_weight)

