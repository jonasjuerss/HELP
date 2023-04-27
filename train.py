import argparse
import json
import os
import traceback
import typing
import warnings
from contextlib import suppress
from datetime import datetime
from functools import partial
from multiprocessing import Process
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch_geometric.loader import DataLoader, DenseDataLoader
from torch_geometric.nn import DenseGCNConv, GCNConv, DenseGINConv
from torchviz import make_dot
from tqdm import tqdm
from typing import Union
import plotly.express as px

import custom_logger
import output_layers
import poolblocks.poolblock
from custom_layers import CustomDenseGINConv
from custom_logger import log
from custom_net import CustomNet
from data_generation.custom_dataset import UniqueMotifCategorizationDataset, CustomDataset, \
    UniqueMultipleOccurrencesMotifCategorizationDataset, UniqueHierarchicalMotifDataset
from data_generation.dataset_wrappers import DatasetWrapper, CustomDatasetWrapper, TUDatasetWrapper, EnzymesWrapper
from data_generation.deserializer import from_dict
from data_generation.motifs import BinaryTreeMotif, HouseMotif, FullyConnectedMotif, CircleMotif
from plot_gradient_flow import plot_grad_flow

DENSE_CONV_TYPES: typing.List[typing.Type[torch.nn.Module]] = [DenseGCNConv, CustomDenseGINConv]
SPARSE_CONV_TYPES = [GCNConv]
VALID_CONV_NAMES = [c.__name__ for c in DENSE_CONV_TYPES + SPARSE_CONV_TYPES]

def train_test_epoch(train: bool, model: CustomNet, optimizer,
                     loader: Union[DataLoader, DenseDataLoader], epoch: int,
                     pooling_loss_weight: float, dense_data: bool, probability_weights_type: str, num_samples: int,
                     mode: str):
    if train:
        model.train()
        sum_loss = 0
        sum_classification_loss = 0
        sum_pooling_loss = 0
        sum_sample_probs = 0
    correct = 0
    num_classes = model.output_layer.num_classes
    if True:  # mode == "val":
        class_counts = torch.zeros(num_classes)
    if epoch == 0:
        class_counts_true = torch.zeros(num_classes)
    with suppress() if train else torch.no_grad():  # nullcontext() would be better here but is not supported on HPC
        for step, data in enumerate(loader):
            data = data.to(custom_logger.device)
            batch_size = data.y.size(0)
            if train:
                optimizer.zero_grad()

            out, probabilities, _, pooling_loss, _ = model(data)
            target = data.y
            if dense_data:
                # For some reason, DataLoader flattens y (e.g. for batch_size=64 and output size 2, it would create one
                # vector of 128 entries). DenseDataLoader doesn't show this behaviour which is why we squeeze in our
                # training loop. As long as we only do graph classification (as opposed to predicting multiple values
                # per graph), we can fix this by just manually introducing the desired dimension with unsqueeze. In the
                # future, we might just use reshape instead of unsqueeze to support multiple output values, but the
                # question, why pytorch geometric behaves this way remains open.
                target = target.squeeze(1)
            if train:
                target = target.repeat(num_samples)
                if probability_weights_type == "none":
                    classification_loss = F.nll_loss(out, target)
                else:
                    assert not probabilities.requires_grad
                    if probability_weights_type == "log_prob":
                        probabilities = torch.log(probabilities)
                    elif probability_weights_type != "prob":
                        raise ValueError(f"Unknown probability weights type {probability_weights_type}!")
                    classification_loss_per_sample = probabilities * F.nll_loss(out, target, reduction='none')
                    sum_sample_probs += torch.sum(probabilities).item()

                    classification_loss = torch.mean(classification_loss_per_sample)
                loss = classification_loss + pooling_loss_weight * pooling_loss + model.custom_losses(batch_size)

                sum_loss += batch_size * float(loss)
                sum_classification_loss += batch_size * float(classification_loss)
                sum_pooling_loss += batch_size * float(pooling_loss)

            pred_classes = out.argmax(dim=1)
            correct += int((pred_classes == target).sum())
            if True: #mode == "val":
                class_counts += torch.bincount(pred_classes.detach(), minlength=num_classes).cpu()
            if epoch == 0:
                class_counts_true += torch.bincount(target, minlength=num_classes).cpu()

            if train:
                # if epoch == 0 and step == 0:
                #     make_dot(loss, dict(model.named_parameters())).render("img/architecture", format="pdf")
                loss.backward()
                # if step == 0 and epoch in [0, 1]:
                #     plot_grad_flow(model.named_parameters(), f"img/gradientflow_{epoch}.pdf")
                optimizer.step()
    dataset_len = len(loader.dataset) * num_samples
    model.log_custom_losses(mode, epoch, dataset_len)
    additional_dict = {}
    if mode == "train":
        additional_dict = {
            f"{mode}_loss": sum_loss / dataset_len,
            f"{mode}_pooling_loss": sum_pooling_loss / dataset_len,
            f"{mode}_classification_loss": sum_classification_loss / dataset_len,
            f"{mode}_avg_sample_prob": sum_sample_probs / dataset_len}
    if True: #elif mode == "val":
        class_counts /= dataset_len
        additional_dict.update({f"{mode}_percentage_class_{i}": class_counts[i] for i in range(num_classes)})
    if epoch == 0:
        class_counts_true /= dataset_len
        additional_dict.update({f"true_{mode}_percentage_class_{i}": class_counts_true[i] for i in range(num_classes)})
    log({f"{mode}_accuracy": correct / dataset_len, **additional_dict},
        step=epoch)
    model.eval()  # make sure model is always in eval by default
    return correct / dataset_len


def log_formulas(model: CustomNet, train_loader: DataLoader, test_loader: DataLoader, class_names: typing.List[str],
                epoch: int):
    model.explain(train_loader, test_loader, class_names)

def process_embeddings(embs, epoch, run):
    tsne = TSNE(n_components=2)
    # pca = PCA(n_components=2)
    for pool_step, emb in enumerate(embs):
        emb = torch.cat(emb, dim=0).detach().numpy()
        coords = tsne.fit_transform(X=emb)
        # for row in range(coords.shape[0]):
        # table.add_data(pool_step, *coords[row], "#000", "")
        fig = px.scatter(x=coords[:, 1], y=coords[:, 0]) # , size=4
        fig.update_traces(marker={'size': 4})
        # path = os.path.join(save_path, f"scatter_{pool_step}.html")
        # fig.write_html(path, auto_play=False)
        # log({f"scatter_{pool_step}": wandb.Html(path)}, step=epoch)
        log({f"embeddings_{pool_step}": fig}, _run=run, step=epoch)

def log_embeddings(model: CustomNet, data_loader: DataLoader, dense_data: bool, epoch: int, save_path):
    # table too big to load (wandb only shows 10000 entries)
    # table = wandb.Table(columns=["pool_step", "x", "y", "point_color", "label"])
    with torch.no_grad():
        if dense_data:
            # list: [num_pool_layers, num_batches] with entries [num_nodes_total_batch, layer_sizes[pool_ste][-1]]
            embs = [[] for _ in model.graph_network.pool_blocks]
            for data in data_loader:
                data.to(custom_logger.device)
                _, _, _, _, pool_activations, _, masks = model(data)
                masks = [data.mask] + masks
                for i, act in enumerate(pool_activations):
                    embs[i].append(act[masks[i]].cpu())
            # list [num_pool_layers] with entries [num_nodes_total, layer_sizes[pool_ste][-1]]
            # TSNE takes some time, so we can let this happen asynchronously
            Process(target=process_embeddings, args=(embs, epoch, wandb.run)).start()

            #log(dict(embeddings=table), step=epoch)
        else:
            print("Logging embeddings not implemented for sparse data yet!")


num_colors = 2
current_dataset = UniqueMultipleOccurrencesMotifCategorizationDataset(BinaryTreeMotif(5, [0], num_colors),
                                                                      [HouseMotif([1], [1], num_colors),
                                                                       FullyConnectedMotif(5, [1], num_colors)],
                                                                      [[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]])
                                                                        # [[0.4, 0.6], [0.4, 0.6]])#

# Note that the colors (and number of colors) here are dummy and will be replaced by the dataset
current_dataset = UniqueHierarchicalMotifDataset([HouseMotif([0], [0], 1),
                                                  FullyConnectedMotif(4, [0], 1),
                                                  CircleMotif(5, [0], 1)],
                                                 [HouseMotif([0], [0], 1),
                                                  FullyConnectedMotif(5, [0], 1),
                                                  FullyConnectedMotif(3, [0], 1)],
                                                 [1/3, 1/3, 1/3],
                                                 [1/3, 1/3, 1/3],
                                                 recolor_lowlevel=True,
                                                 randomize_colors=True,
                                                 one_hot_color=True,
                                                 insert_intermediate_nodes=True,
                                                 perturb=0.0)

current_dataset_wrapper = CustomDatasetWrapper(current_dataset)
# current_dataset_wrapper = EnzymesWrapper()
def parse_json_str(s: str):
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ["\"", "'"]:
        s = s[1:-2] # remove possible quotation marks around whole json
    return json.loads(s)

def main(args, **kwargs):
    if not isinstance(args, dict):
        args = args.__dict__
    restore_path = None
    if args["resume"] is not None:
        api = wandb.Api()
        run_path = f"{custom_logger.wandb_entity}/{custom_logger.wandb_project}/" + args["resume"]
        run = api.run(run_path)
        save_path = args["save_path"]
        args = run.config
        for k, v in kwargs.items():
            args[k] = v
        restore_path = args["save_path"] + f"/checkpoint{'_last' if args['resume_last'] else ''}.pt"
        if args["save_wandb"] and not os.path.isfile(restore_path):
            print("Downloading checkpoint from wandb...")
            wandb.restore(restore_path, run_path=run_path)
        args["save_path"] = save_path
    else:
        if not args["use_wandb"] and args["save_wandb"]:
            print("Disabling saving to wandb as logging to wandb is also disabled.")
            args["save_wandb"] = False

    if isinstance(args, dict):
        args = SimpleNamespace(**args)

    if "dummy" in args.save_path:
        pass
    elif os.path.exists(args.save_path):
        raise ValueError(f"Checkpoint path already exists: {args.save_path}!")
    else:
        os.makedirs(args.save_path)

    if args.probability_weights != "none" and any([block_args.get("soft_sampling", -1) == 0
                                                   for block_args in args.pool_block_args]):
        warnings.warn("Cluster assignments are deterministic. Setting probability weights to none.")
        args.probability_weights = "none"

    for i, block_args in enumerate(args.pool_block_args):
        if args.pooling_type[i] in ["ASAP"] and block_args.get("num_output_nodes", -1) > args.min_nodes:
            print(f"The pooling method {args.pooling_type} cannot increase the number of nodes. Increasing "
                  f"min_nodes to {block_args['num_output_nodes']} to guarantee the given fixed number of output nodes.")
            args.min_nodes = block_args["num_output_nodes"]
    args = custom_logger.init(args)

    device = torch.device(args.device)
    custom_logger.device = device
    custom_logger.cpu_workers = args.cpu_workers


    dataset_wrapper = typing.cast(DatasetWrapper, from_dict(args.dataset))
    torch.manual_seed(0)  # Ensure deterministic dataset generation/split
    np.random.seed(0)
    dataset = dataset_wrapper.get_dataset(args.dense_data, args.min_nodes)
    num_train_samples = int(args.train_split * len(dataset))
    num_val_samples = int(args.val_split * len(dataset))
    train_data = dataset[:num_train_samples]
    val_data = dataset[num_train_samples:num_train_samples + num_val_samples]
    test_data = dataset[num_train_samples + num_val_samples:]
    torch.manual_seed(args.seed)

    if args.dense_data:
        train_loader = DenseDataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        test_loader = DenseDataLoader(test_data, batch_size=args.batch_size, shuffle=True)
        val_loader = DenseDataLoader(val_data, batch_size=args.batch_size, shuffle=True)
        log_graph_loader = DenseDataLoader(test_data[:args.batch_size], batch_size=args.batch_size, shuffle=False)

    else:
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True)
        log_graph_loader = DataLoader(test_data[:args.batch_size], batch_size=args.batch_size, shuffle=False)

    # Get last (and only data batch from log_graph_loader)
    for graphs_to_log in log_graph_loader:
        pass

    CONV_TYPES = DENSE_CONV_TYPES if args.dense_data else SPARSE_CONV_TYPES
    conv_type = next((x for x in CONV_TYPES if x.__name__ == args.conv_type), None)
    if conv_type is None:
        raise ValueError(f"No convolution type named \"{args.conv_type}\" found for dense_data={args.dense_data}!")
    output_layer = output_layers.from_name(args.output_layer)
    gnn_activation = getattr(torch.nn.functional, args.gnn_activation)
    pooling_block_types = [poolblocks.poolblock.from_name(pt, args.dense_data) for pt in args.pooling_type]
    model = CustomNet(dataset_wrapper.num_node_features, dataset_wrapper.num_classes, args=args, device=device,
                      output_layer_type=output_layer,
                      pooling_block_types=pooling_block_types,
                      conv_type=conv_type, activation_function=gnn_activation,
                      directed_graphs=dataset_wrapper.is_directed)
    if restore_path is not None:
        model.load_state_dict(torch.load(restore_path))
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    max_val_acc = 0
    model_save_path_best = args.save_path + "/checkpoint.pt"
    model_save_path_last = args.save_path + "/checkpoint_last.pt"
    save_same_acc_cooldown = 20
    last_best_save = -save_same_acc_cooldown
    if args.blackbox_transparency != 1:
        num_samples = 1
    else:
        num_samples = np.prod([pb_arg.get("num_mc_samples", 1) for pb_arg in args.pool_block_args]).item()
    for epoch in tqdm(range(args.num_epochs)):
        train_test_epoch(True, model, optimizer, train_loader,
                         epoch, args.pooling_loss_weight, args.dense_data, args.probability_weights, num_samples,
                         "train")
        test_acc = train_test_epoch(False, model, optimizer, test_loader, epoch,
                                    args.pooling_loss_weight, args.dense_data, args.probability_weights, 1,
                                    "test")
        val_acc = train_test_epoch(False, model, optimizer, val_loader, epoch,
                                   args.pooling_loss_weight, args.dense_data, args.probability_weights, 1,
                                   "val")

        try:
            if epoch % args.graph_log_freq == 0:
                model.graph_network.pool_blocks[0].log_assignments(model, graphs_to_log, args.graphs_to_log, epoch)
            for i, block in enumerate(model.graph_network.pool_blocks):
                block.log_data(epoch, i)

            if epoch % args.formula_log_freq == 0:
                log_formulas(model, train_loader, test_loader, dataset_wrapper.class_names, epoch)
            if val_acc > max_val_acc or (
                    val_acc == max_val_acc and last_best_save + save_same_acc_cooldown <= epoch):
                print(
                    f"Saving model with validation accuracy {100 * val_acc:.2f}% (test accuracy {100 * test_acc:.2f}%)")
                torch.save(model.state_dict(), model_save_path_best)
                if args.save_wandb:
                    wandb.save(model_save_path_best, policy="now")
                max_val_acc = val_acc
                last_best_save = epoch
                log({"best_val_acc": max_val_acc, "test_at_best_val_acc": test_acc}, step=epoch)
            if epoch % args.save_freq == 0:
                torch.save(model.state_dict(), model_save_path_last)
                if args.save_wandb:
                    wandb.save(model_save_path_last, policy="now")
        except Exception:
            print("Error occurred while logging:")
            traceback.print_exc()
            # log_embeddings(model, train_loader, args.dense_data, epoch, args.save_path)
        model.end_epoch()
    if args.num_epochs > 0:
        if args.graph_log_freq >= 0:
            model.graph_network.pool_blocks[0].log_assignments(model, graphs_to_log, args.graphs_to_log, epoch)
        if args.formula_log_freq >= 0:
            log_formulas(model, train_loader, test_loader, dataset_wrapper.class_names, epoch)
    return model, args, dataset_wrapper, train_loader, val_loader, test_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.01,
                        help='The Adam learning rate to use.')
    parser.add_argument('--pooling_loss_weight', type=float, default=0.5,
                        help='The weight of the pooling loss.')
    parser.add_argument('--entropy_loss_weight', type=float, default=0,
                        help='The weight of the entropy loss in the explanation layer.')
    parser.add_argument('--wd', type=float, default=5e-4,
                        help='The Adam weight decay to use.')
    parser.add_argument('--blackbox_transparency', type=float, default=1,
                        help='Transparency of the hyperplane gradient approximation. If 1, no hyperplane estimation is '
                             'used, if 0, only hyperplane estimation is used and for values in between, the gradients '
                             'are given as a weighted sum.')
    parser.add_argument('--num_epochs', type=int, default=20000,
                        help='The number of epochs to train for.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='The batch size to use.')
    parser.add_argument('--add_layer', type=int, nargs='+', action='append',
                        default=[], dest='layer_sizes',
                        help='The layer sizes to use. Example: --add_layer 16 32 --add_layer 32 64 16 results in a '
                             'network with 2 pooling steps where 5 message passes are performed before the first and ')
    parser.add_argument('--pool_blocks', type=parse_json_str, nargs='+',
                        # default=[{"num_output_layers": [4]}],
                        # default=[{"num_output_nodes": 8}],
                        # default=[{"num_concepts": 3}],
                        # default=[],
                        dest='pool_block_args',
                        help="Additional arguments for each pool block")
    # parser.add_argument('--nodes_per_layer', type=int, default=[4],
    #                     help='The number of nodes after each pooling step for architectures like DiffPool which require'
    #                          ' to pre-specify that. Note that the last one should be 1 for classification')

    parser.add_argument('--conv_type', type=str, default="DenseGCNConv", choices=VALID_CONV_NAMES,
                        help='The type of graph convolution to use.')
    parser.add_argument('--output_layer', type=str, default="DenseClassifier",
                        help='The type of graph convolution to use.')
    # TODO sum might be too weak, implement Pietro's global pool
    parser.add_argument('--output_layer_merge', type=str, default="flatten", choices=["flatten", "none", "sum", "avg"],
                        help='How to merge the output encodings of all nodes after the last pooling step for the final '
                             'classification layer. \"flatten\" only works if the number of clusters in the last graph '
                             'is constant/independent of the input graph size and \"none\" only if the chosen '
                             'classifier can deal with a set of inputs.')
    parser.add_argument('--pooling_type', type=str, default="Perturbed", choices=poolblocks.poolblock.valid_names(),
                        nargs='+',
                        help='The type of pooling to use.')

    parser.add_argument('--dataset', type=parse_json_str, default=current_dataset_wrapper.__dict__(),
                        help="A json that defines the current dataset")
    parser.add_argument('--min_nodes', type=int, default=0,
                        help='Minimum number of nodes for a graph in the dataset. All other graphs are discarded. '
                             'Required e.g. to guarantee that ASAPooling always outputs a fixed number of nodes when '
                             'num_output_nodes is set.')

    parser.add_argument('--train_split', type=float, default=0.8,
                        help='Fraction of samples used for the train set.')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Fraction of samples used for the validation set.')

    parser.add_argument('--graph_log_freq', type=int, default=50,
                        help='Every how many epochs to log graphs to wandb. The final predictions will always be '
                             'logged, except for if this is negative.')
    parser.add_argument('--formula_log_freq', type=int, default=50,
                        help='Every how many epochs to log explanations to wandb. The final predictions will always be '
                             'logged, except for if this is negative.')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='Every how many epochs to save a checkpoint. This is in addition to the best checkpoint of'
                             ' highest validation accuracy.')
    parser.add_argument('--graphs_to_log', type=int, default=6,
                        help='How many graphs from the training and testing set to log.')
    parser.add_argument('--forced_embeddings', type=float, default=None,
                        help='For debugging. If set, embeddings will not be calculated. Instead, all embeddings of '
                             'nodes with neighbours will be set to the given number and all nodes without neighbours '
                             'will have embedding 0.')
    # TODO should I use the log probs instead?
    parser.add_argument('--probability_weights', type=str, default="none", choices=["none", "prob", "log_prob"],
                        help='The loss contribution of each sample can be weighted by the (log) probability of all '
                             'sampled cluster assignments (REINFORCE).')
    parser.add_argument('--gnn_activation', type=str, default="leaky_relu",
                        help='Activation function to be used in between the GNN layers')

    parser.set_defaults(dense_data=True)
    parser.add_argument('--sparse_data', action='store_false', dest='dense_data',
                        help='Switches from a dense representation of graphs (dummy nodes are added so that each of '
                             'them has the same number of nodes) to a sparse one.')
    parser.add_argument('--dense_data', action='store_true', dest='dense_data',
                        help='Switches from a sparse data representation of graphs to a dense one (dummy nodes are '
                             'added so that each of them has the same number of nodes).')

    parser.add_argument('--seed', type=int, default=1,
                        help='The seed used for pytorch.')
    parser.add_argument('--cpu_workers', type=int, default=0,
                        help='How many workers to spawn for cpu parallelization (like for clustering). '
                             'If 0, everything will happen in the main process.')
    parser.add_argument('--save_path', type=str,
                        default=os.path.join("models", datetime.now().strftime("%d-%m-%Y_%H-%M-%S")),
                        help='The path to save the checkpoint to. Will be models/dd-mm-YY_HH-MM-SS.pt by default.')
    parser.add_argument('--device', type=str, default="cuda",
                        help='The device to train on. Allows to use CPU or different GPUs.')
    parser.set_defaults(use_wandb=True)
    parser.add_argument('--no_wandb', action='store_false', dest='use_wandb',
                        help='Turns off logging to wandb')
    parser.add_argument('--save_wandb', action='store_true', help="Whether to upload the checkpoint files to wandb.")
    parser.add_argument('--no-save_wandb', dest='save_wandb', action='store_false')
    parser.set_defaults(save_wandb=True)
    parser.add_argument('--wandb_name', type=str, default=None,
                        help="Name of the wandb run. Standard randomly generated wandb names if not specified.")
    parser.add_argument('--resume', type=str, default=None,
                        help='Will load configuration from the given wandb run and load the locally stored weights.')
    parser.add_argument('--resume_last', action='store_true', help="Whether to resume from the last checkpoint."
                                                                   "By default resuming from the best one.")
    parser.add_argument('--resume_best', dest='resume_last', action='store_false')
    parser.set_defaults(resume_last=False)

    main(parser.parse_args())

