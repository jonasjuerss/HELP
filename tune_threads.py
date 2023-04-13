import timeit

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

import custom_logger

if __name__ == "__main__":
    runtimes = []
    threads = [1] + [t for t in range(2, 49, 2)]
    for t in threads:
        torch.set_num_threads(t)
        r = timeit.timeit(setup="import torch; x = torch.randn(1024, 1024); y = torch.randn(1024, 1024)", stmt="torch.mm(x, y)", number=100)
        runtimes.append(r)

    fig, ax = plt.subplots()
    ax.plot(np.array(threads), np.array(runtimes))
    fig.savefig("img/threads.pdf")


    wandb_data = [[x, y] for (x, y) in zip(threads, runtimes)]
    table = wandb.Table(data=wandb_data, columns=["threads", "runtime"])
    wandb.init(project=custom_logger.wandb_project, entity=custom_logger.wandb_entity, dir="/tmp/thesis",
               name="profiling")
    wandb.log({"threads": wandb.plot.line(table, "threads", "runtime", title="Number of threads vs runtime")})
    # wandb.log({"threads": fig})