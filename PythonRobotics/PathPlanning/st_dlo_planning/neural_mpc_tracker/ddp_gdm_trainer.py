'''
    Train GDM Script
'''
import torch
import pathlib
from omegaconf import OmegaConf
import dill
import copy
import glob
import os

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

import torch.nn as nn
from torch.utils.data import DataLoader, random_split, ConcatDataset, DistributedSampler
import torch.optim as optim

from st_dlo_planning.utils.pytorch_utils import dict_apply, to_numpy, optimizer_to
from st_dlo_planning.utils.misc_utils import setup_seed
from st_dlo_planning.neural_mpc_tracker.gdm_dataset import MultiStepGDMDataset
from st_dlo_planning.neural_mpc_tracker.modelling_gdm import GDM, GDM_CFG
from st_dlo_planning.neural_mpc_tracker.base_trainer import BaseTrainer, CosineWarmupScheduler, DataLogger


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def prepare_dataloader(dataset, rank, world_size, batch_size, pin_memory=False, num_workers=0):
    '''
        Prepare dataloader for distributed training'''
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=False, shuffle=False, sampler=sampler)
    return dataloader


class DDPGDMTrainer(BaseTrainer):
    include_keys = ('epoch',) 
    exclude_keys = tuple() 

    def __init__(self, train_cfg: OmegaConf, model_cfg):
        '''
            global deformation model trainer
        '''
        super().__init__(train_cfg)

        # configure model
        self.model = GDM(model_cfg)

        # configure optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), 
                                           lr=train_cfg.optimizer.lr, 
                                           betas=train_cfg.optimizer.betas,
                                           weight_decay=train_cfg.optimizer.weight_decay)
        # lr scheduler
        self.scheduler = CosineWarmupScheduler(optimizer=self.optimizer, 
                                               warmup=train_cfg.train.lr_warmup_steps, 
                                               max_iters=train_cfg.train.num_epochs)
        self.epoch = 0


    def run_batch(self, loss_fn, batch):
        '''
            training module
        '''
        dlo_keypoints = batch['dlo_keypoints']
        eef_states = batch['eef_states']

        delta_eef = batch['delta_eef']
        delta_shape = batch['delta_shape']

        predicted_delta_shape = self.model(dlo_keypoints, eef_states, delta_eef)

        loss = loss_fn(predicted_delta_shape, delta_shape)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.parameters()), self._cfg.optimizer.grad_clip)
        self.optimizer.step()
        l_np = to_numpy(loss)
        return l_np
    

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        ckpt_path = f"{epoch}.ckpt"
        torch.save(ckp, ckpt_path)
        print(f"Epoch {epoch} | Training checkpoint saved at {ckpt_path}")
    

    def run(self, rank, world_size):
        '''
            training module
        '''
        cfg = copy.deepcopy(self._cfg)
        ddp_setup(rank, world_size)
        self.model = self.model.to(rank)
        self.model = DDP(self.model, device_ids=[rank])

        train_batch_size = cfg.train_dataloader.batch_size // world_size
        max_epoch = cfg.train.num_epochs

        # =================== configure train & validation dataset & test dataset ===================
        train_data_dirs = cfg.train_dataloader.data_dir
        train_datasets = []
        for sub_data_dir in train_data_dirs:
            train_data_paths = glob.glob(os.path.join(sub_data_dir, 'task*.zarr'))
            for train_data_path in train_data_paths:
                sub_dataset = MultiStepGDMDataset(data_path=train_data_path, max_step=30)
                train_datasets.append(sub_dataset)
        data = ConcatDataset(train_datasets)
        data_size = data.cumulative_sizes[-1]

        train_size = int(0.8 * data_size)
        val_size = data_size - train_size
        print(f'Training Data Size: {train_size} Validation Datasize: {val_size}')
        train_dataset, _ = random_split(data, [train_size, val_size])

        # =================== configure train & validation & test dataloader ===================
        train_dataloader = prepare_dataloader(train_dataset, rank, world_size, train_batch_size)
        
        # configure logging
        if rank == 0:
            wandb_logger = DataLogger(log_dir=cfg.logging.output_dir, wandb_entity='tangyunxi000', config=cfg)

        # configure loss function
        loss_fn = nn.MSELoss(reduction='mean')

        # ========================= loop ===========================
        for epoch_idx in range(max_epoch):
            train_dataloader.sampler.set_epoch(epoch_idx)
            # train
            train_loss = 0
            total_samples = 0
            for ele in train_dataloader:
                l_np = self.run_batch(loss_fn, ele)
                train_loss += l_np
                total_samples += ele['dlo_keypoints'].shape[0]
            
            
            if rank == 0:
                wandb_logger.log("train_loss", train_loss, self.epoch)
                wandb_logger.log("ave_train_loss", train_loss / total_samples, self.epoch)
                wandb_logger.log("lr", self.scheduler.get_lr()[0], self.epoch)
            

            if cfg.checkpoint.save_checkpoint and epoch_idx % cfg.checkpoint.checkpoint_every == 0 and rank == 0:
                self._save_checkpoint(tag=f'epoch_{self.epoch}')

            self.scheduler.step()
            self.epoch += 1

        if cfg.checkpoint.save_last_ckpt and rank == 0:
            self.save_cfg()
            self._save_checkpoint()
        if rank == 0:
            wandb_logger.close()
        destroy_process_group()
        return None


def main_worker(rank, world_size, train_cfg, model_cfg):
    trainer = DDPGDMTrainer(train_cfg, model_cfg)
    trainer.run(rank, world_size)


if __name__ == "__main__":
    train_cfg = OmegaConf.load('path_to_config.yaml')
    gdm_cfg = GDM_CFG()
    world_size = torch.cuda.device_count()

    mp.spawn(main_worker, 
             args=(world_size, train_cfg, gdm_cfg), 
             nprocs=world_size, join=True)