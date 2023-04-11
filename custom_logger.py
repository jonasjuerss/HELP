from typing import Optional

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
            dir="/tmp/thesis",
            config=args
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
