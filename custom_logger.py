from typing import Optional, Any, Dict, List

import wandb
device = None
cpu_workers: Optional[int] = None
wandb_project = "MPhil-project"
wandb_entity = "jonas-juerss"


def init(args):
    if args.use_wandb:
        wandb_args = dict(
            project=wandb_project,
            entity=wandb_entity,
            dir=args.wandb_dir if hasattr(args, "wandb_dir") else "wandb",
            config=args,
            tags=[args.dataset["_type"][:-7]] + (args.wandb_tags if hasattr(args, "wandb_tags") else [])
        )
        if args.wandb_name is not None:
            wandb_args["name"] = args.wandb_name
        wandb.init(**wandb_args)
        return wandb.config
    return args


def log(*args, _run=None, **kwargs):
    run = wandb.run if _run is None else _run
    if run is not None:
        run.log(*args, **kwargs)


def plot_table(vega_spec_name: str, data_table: wandb.Table, fields: Dict[str, Any],
               string_fields: Optional[Dict[str, Any]] = None, _run=None, **kwargs):
    """Creates a custom plot on a table.

    Arguments:
        vega_spec_name: the name of the spec for the plot
        data_table: a wandb.Table object containing the data to
            be used on the visualization
        fields: a dict mapping from table keys to fields that the custom
            visualization needs
        string_fields: a dict that provides values for any string constants
            the custom visualization needs
    """
    run = wandb.run if _run is None else _run
    if run is not None:
        return run.plot_table(vega_spec_name, data_table, fields, string_fields, **kwargs)
