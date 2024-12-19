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

import torch.nn as nn
from torch.utils.data import DataLoader, random_split, ConcatDataset
import torch.optim as optim

from st_dlo_planning.utils.pytorch_utils import dict_apply, to_numpy, optimizer_to
from st_dlo_planning.utils.misc_utils import setup_seed
from st_dlo_planning.neural_mpc_tracker.gdm_dataset import MultiStepGDMDataset
from st_dlo_planning.neural_mpc_tracker.modelling_gdm import GDM, GDM_CFG
from st_dlo_planning.neural_mpc_tracker.base_trainer import BaseTrainer, CosineWarmupScheduler, DataLogger

import numpy as np


class GDMTrainer(BaseTrainer):
    include_keys = ('epoch',) 
    exclude_keys = tuple() 

    def __init__(self, cfg: OmegaConf, gdm_cfg: GDM_CFG):
        '''
            global deformation model trainer
        '''
        super().__init__(cfg)

        # set seed for reproduction
        seed = cfg.train.seed
        setup_seed(seed)

        # configure model
        self.model = GDM(gdm_cfg)

        # configure optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), 
                                           lr=cfg.optimizer.lr, 
                                           betas=cfg.optimizer.betas,
                                           weight_decay=cfg.optimizer.weight_decay)
        # lr scheduler
        self.scheduler = CosineWarmupScheduler(optimizer=self.optimizer, 
                                               warmup=cfg.train.lr_warmup_steps, 
                                               max_iters=cfg.train.num_epochs)
        self.epoch = 0


    def train_step(self, loss_fn, batch):
        '''
            train step for one batch
        '''
        self.model.train()
        state = batch['state']
        delta_ee_pos = batch['delta_ee_pos']
        delta_shape = batch['delta_shape']
        dlo_len = batch['dlo_len']

        predicted_delta_shape = self.model(state, delta_ee_pos, dlo_len)
        loss = loss_fn(delta_shape, predicted_delta_shape)

        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.parameters()), 
        #                                1.0)
        self.optimizer.step()

        l_np = to_numpy(loss)
        
        return l_np
    
    def validate_step(self, loss_fn, batch):
        self.model.eval()
        state = batch['state']
        delta_ee_pos = batch['delta_ee_pos']
        delta_shape = batch['delta_shape']
        dlo_len = batch['dlo_len']

        predicted_delta_shape = self.model(state, delta_ee_pos, dlo_len)
        loss = loss_fn(delta_shape, predicted_delta_shape)

        l_np = to_numpy(loss)
        
        return l_np
    
    def run(self):
        '''
            training module
        '''
        cfg = copy.deepcopy(self._cfg)
        device = torch.device(cfg.train.device)
        train_batch_size = cfg.train_dataloader.batch_size
        val_batch_size = cfg.val_dataloader.batch_size
        max_epoch = cfg.train.num_epochs

        # device transfer
        self.model.to(device)
        optimizer_to(self.optimizer, device)

        # =================== configure train & validation dataset & test dataset ===================
        train_data_dirs = cfg.train_dataloader.data_dir
        
        train_datasets = []
        for sub_data_dir in train_data_dirs:
            train_data_paths = glob.glob(os.path.join(sub_data_dir, '*_train.zarr'))
            for train_data_path in train_data_paths:
                sub_dataset = MultiStepGDMDataset(data_path=train_data_path)
                train_datasets.append(sub_dataset)
        data = ConcatDataset(train_datasets)
        data_size = data.cumulative_sizes[-1]

        train_size = int(0.8 * data_size)
        val_size = data_size - train_size
        print(f'Training Data Size: {train_size} Validation Datasize: {val_size}')
        train_dataset, val_dataset = random_split(data, [train_size, val_size])

        # =================== configure train & validation & test dataloader ===================
        train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        
        val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
        
        # configure logging
        wandb_logger = DataLogger(
            log_dir='/home/yxtang/CodeBase/DOBERT/outputs',
            wandb_entity='tangyunxi000',
            config=cfg
        )

        # configure loss function
        loss_fn = nn.MSELoss(reduction='mean')

        # ========================= loop ===========================
        for epoch_idx in range(max_epoch):
            # train
            train_loss = 0
            total_samples = 0
            for ele in train_dataloader:
                ele = dict_apply(ele, func= lambda x: x.to(device))
                l_np = self.train_step(loss_fn, ele)
                train_loss += l_np
                total_samples += ele['dlo_len'].shape[0]
            wandb_logger.log("train_loss", train_loss, self.epoch)
            wandb_logger.log("ave_train_loss", train_loss / total_samples, self.epoch)
            wandb_logger.log("lr", self.scheduler.get_lr()[0], self.epoch)
            
            # validation
            if epoch_idx % cfg.train.validation_every == 0:
                with torch.no_grad():
                    val_loss = 0
                    total_samples = 0
                    for ele in val_dataloader:
                        ele = dict_apply(ele, func= lambda x: x.to(device))
                        l_np = self.validate_step(loss_fn, ele)
                        val_loss += l_np
                        total_samples += ele['dlo_len'].shape[0]
                print(f'Epoch {self.epoch}: Train Loss {train_loss} || Val Loss {val_loss}')
                wandb_logger.log("val_loss", val_loss, self.epoch)
                wandb_logger.log("ave_val_loss", val_loss / total_samples, self.epoch)

            if cfg.checkpoint.save_checkpoint and epoch_idx % cfg.checkpoint.checkpoint_every == 0:
                self.save_checkpoint(tag=f'epoch_{self.epoch}')

            self.scheduler.step()
            self.epoch += 1

        if cfg.checkpoint.save_last_ckpt:
            self.save_cfg()
            self.save_checkpoint()
        wandb_logger.close()
        return None