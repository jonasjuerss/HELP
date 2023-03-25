import wandb
device = None
def init(args):
    if args.use_wandb:
        wandb.init(project="MPhil-project", entity="jonas-juerss", config=args)
        return wandb.config
    return args

def log(*args, _run=None, **kwargs):
    run = wandb.run if _run is None else _run
    if run is not None:
        run.log(*args, **kwargs)
