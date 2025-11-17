import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import OrderedDict
import dill
import logging


def ddp_setup(world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    dist.init_process_group(backend="nccl", world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def save_checkpoint(model, optimizer, ema, ckpt_path: str):
    checkpoint = {
        "model": model.module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "ema": ema.state_dict() if ema is not None else None,
    }
    torch.save(checkpoint, ckpt_path, pickle_module=dill)


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)
    return None


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"{logging_dir}/log.txt", mode="w"),
            ],
        )
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def sync_metrics(metric_tensor, world_size):
    """
    Synchronize a metric across all processes using all_reduce. Returns the averaged value
    """
    dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
    return metric_tensor.item() / world_size


class MyDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.data = torch.arange(0, 16, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index : index + 1]


class ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(1, 1)

    def forward(self, x):
        return self.layer(x)


def main(world_size, num_epoch):
    ddp_setup(world_size)
    rank = dist.get_rank()
    pid = os.getpid()

    device_id = rank % torch.cuda.device_count()

    print(f"current pid: {pid}, current rank: {rank} device_id: {device_id}")

    model = ToyModel().to(device_id)
    ddp_model = DDP(model, device_ids=[device_id])
    loss_fn = nn.MSELoss()
    optimizer = optim.AdamW(ddp_model.parameters(), lr=0.001)

    dataset = MyDataset()
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    dataloader = DataLoader(dataset, batch_size=2, sampler=sampler)

    if rank == 0:
        for key, val in ddp_model.module.state_dict().items():
            print(rank, key, val)

        for key, val in optimizer.state_dict().items():
            print(key, val)
    dist.barrier()

    logger = create_logger(logging_dir=".")

    train_step = 0
    for epoch in range(num_epoch):
        sampler.set_epoch(epoch)
        epoch_loss = 0.0
        for x in dataloader:
            x = x.to(device_id)
            y = ddp_model(x)
            loss = loss_fn(x, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss = epoch_loss + loss.item()

            avg_loss = sync_metrics(loss, world_size)
            logger.info(f"train step: {train_step} loss: {avg_loss}")

            train_step += 1

            if epoch % 20 == 0:
                if rank == 0:
                    ckpt_path = f"checkpoint_{epoch}.ckpt"
                    save_checkpoint(ddp_model, optimizer, ema=None, ckpt_path=ckpt_path)
                dist.barrier()

    cleanup()


if __name__ == "__main__":
    # mp.spawn(main, args=(8, 10), nprocs=8)
    main(world_size=8, num_epoch=200)
