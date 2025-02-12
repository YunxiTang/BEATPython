from typing import Optional, Dict
import os
import pathlib
import copy
from omegaconf import OmegaConf
import dill
import yaml
import torch
import threading
from typing import Callable
import numpy as np
import torch.optim as optim
import wandb
import PIL


def to_numpy(tensor):
    """
        convert a tensor to numpy variable
    """
    return tensor.to('cpu').detach().numpy()


def dict_apply(x: dict, func: Callable) -> dict:
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result


def _copy_to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().to('cpu')
    elif isinstance(x, dict):
        result = dict()
        for k, v in x.items():
            result[k] = _copy_to_cpu(v)
        return result
    elif isinstance(x, list):
        return [_copy_to_cpu(k) for k in x]
    else:
        return copy.deepcopy(x)



class TopKCheckpointManager:
    def __init__(self, 
                 save_dir, 
                 monitor_key: str, 
                 mode='min', k=1, 
                 format_str='epoch={epoch:03d}-train_loss={train_loss:.3f}.ckpt'
        ):
        assert mode in ['max', 'min']
        assert k >= 0

        self.save_dir = save_dir
        self.monitor_key = monitor_key
        self.mode = mode
        self.k = k
        self.format_str = format_str
        self.path_value_map = dict()
    
    def get_ckpt_path(self, data: Dict[str, float]) -> Optional[str]:
        if self.k == 0:
            return None

        value = data[self.monitor_key]
        ckpt_path = os.path.join(self.save_dir, self.format_str.format(**data))
        
        if len(self.path_value_map) < self.k:
            # under-capacity
            self.path_value_map[ckpt_path] = value
            return ckpt_path
        
        # at capacity
        sorted_map = sorted(self.path_value_map.items(), key=lambda x: x[1])
        min_path, min_value = sorted_map[0]
        max_path, max_value = sorted_map[-1]

        delete_path = None
        if self.mode == 'max':
            if value > min_value:
                delete_path = min_path
        else:
            if value < max_value:
                delete_path = max_path

        if delete_path is None:
            return None
        else:
            del self.path_value_map[delete_path]
            self.path_value_map[ckpt_path] = value

            if not os.path.exists(self.save_dir):
                os.mkdir(self.save_dir)

            if os.path.exists(delete_path):
                os.remove(delete_path)
            return ckpt_path


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, max_iters, min_lr:float=1e-5):
        self.warmup = warmup
        self.max_num_iters = max_iters
        self.min_lr = min_lr
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [np.maximum(base_lr * lr_factor, self.min_lr) for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


class DataLogger:
    """
        Data Logging class to log metrics to wandb/tensorboard 
        (with retrieve running statistics about logged data as an optional).
    """
    include_cfg_keys = ('meta', 'train')
    
    def __init__(self, 
                 log_dir: str, 
                 config=None, 
                 wandb_entity=None, 
                 wandb_mode: str = 'online', 
                 log_wandb=True, 
                 log_tb=False):
        """
        Args:
            log_dir (str): base path to store logs
            log_tb (bool): whether to use tensorboard logging
        """
        self._tb_logger = None
        self._wandb_logger = None
        self._data = dict() 

        if log_tb:
            from tensorboardX import SummaryWriter
            self._tb_logger = SummaryWriter(os.path.join(log_dir), flush_secs=1, max_queue=1)

        if log_wandb:
            self._wandb_logger = wandb.init(entity=wandb_entity,
                                            project=config.logging.wandb_project,
                                            name=config.logging.experiment_name + f'_{config.train.seed}',
                                            dir=log_dir,
                                            mode=wandb_mode,
                                            )

            # set up info experiment identification (meta + train + model)
            wandb_config = dict()
            for key in self.include_cfg_keys:
                for (k, v) in config[key].items():
                    wandb_config[k] = v

            wandb.config.update(wandb_config)
        

    def log(self, k, v, epoch, data_type='scalar', log_stats=False):
        """
        Record data with logger.
        Args:
            k (str): key string
            v (float or image): value to store
            epoch: current epoch number
            data_type (str): the type of data. either 'scalar' or 'image'
            log_stats (bool): whether to store the mean/max/min/std for all logged data with key k
        """

        assert data_type in ['scalar', 'image']

        if data_type == 'scalar':
            if log_stats or k in self._data: 
                if k not in self._data:
                    self._data[k] = []
                self._data[k].append(v)

        # log to tensorboardX
        if self._tb_logger is not None:
            if data_type == 'scalar':
                self._tb_logger.add_scalar(k, v, epoch)

                if log_stats:
                    stats = self.get_stats(k)
                    for (stat_k, stat_v) in stats.items():
                        stat_k_name = '{}-{}'.format(k, stat_k)
                        self._tb_logger.add_scalar(stat_k_name, stat_v, epoch)

            elif data_type == 'image':
                self._tb_logger.add_images(k, img_tensor=v, global_step=epoch, dataformats="NHWC")

        # log to wandb
        if self._wandb_logger is not None:
            try:
                if data_type == 'scalar':
                    self._wandb_logger.log({k: v}, step=epoch)

                    if log_stats:
                        stats = self.get_stats(k)
                        for (stat_k, stat_v) in stats.items():
                            self._wandb_logger.log(
                                {"{}/{}".format(k, stat_k): stat_v}, 
                                step=epoch
                                )

                elif data_type == 'image':
                    image = PIL.Image.fromarray(v)
                    self._wandb_logger.log(
                        {k: wandb.Image(image)},
                        step=epoch
                    )
                
            except Exception as e:
                print("wandb logging: {}".format(e))

    def get_stats(self, k):
        """
        Computes running statistics for a particular key.
        Args:
            k (str): key string
        Returns:
            stats (dict): dictionary of statistics
        """
        stats = dict()
        stats['mean'] = np.mean(self._data[k])
        stats['std'] = np.std(self._data[k])
        stats['min'] = np.min(self._data[k])
        stats['max'] = np.max(self._data[k])
        return stats


    def close(self):
        """
            Run before terminating to make sure all logs are flushed
        """
        if self._tb_logger is not None:
            self._tb_logger.close()

        if self._wandb_logger is not None:
            self._wandb_logger.finish()


class BaseTrainer:
    include_keys = tuple()
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf):
        self._cfg = cfg
        self._saving_thread = None
        self._ckpt_dir = cfg.checkpoint.ckpt_dir
        self._output_dir = cfg.logging.output_dir

    @property
    def output_dir(self):
        return self._output_dir
    
    def run(self):
        """
            Create all the resource as local variables to avoid being serialized,

            which means no self.xxx variables should be created here.
        """
        # resume training or create new training trial

        # configure dataset

        # configure validation dataset

        # configure optimizer

        # configure logging

        # configure checkpoint

        # device transfer

        # training loop
            # ========= train for each epoch ==========
            # ========= eval for at certain epoch =====
            # ========= save checkpoint ===============
            # ========= log at end of epoch ===========
        
        # save latest checkpoint
        pass

    def save_cfg(self, cfg_path=None):
        '''
            save the training config file as `cfg.yaml` into checkpoint
        '''
        if cfg_path is None:
            cfg_path = os.path.join(self._cfg.checkpoint.ckpt_dir, f'cfg.yaml')
        cfg_path = pathlib.Path(cfg_path)
        cfg_path.parent.mkdir(parents=False, exist_ok=True)
        OmegaConf.save(self._cfg, cfg_path)
        return str(cfg_path.absolute())
    
    def save_model_cfg(self, model_cfg_path=None):
        if model_cfg_path is None:
            model_cfg_path = os.path.join(self._cfg.checkpoint.ckpt_dir, f'model_cfg.yaml')
        model_cfg_path = pathlib.Path(model_cfg_path)
        model_cfg_path.parent.mkdir(parents=False, exist_ok=True)
        OmegaConf.save(self._cfg.model, model_cfg_path)
        return str(model_cfg_path.absolute())


    def save_data_stats(self, datastats: dict, datastats_path = None):
        '''
            save training dataset stats as `data_stats.yaml` file into checkpoint
        '''
        if datastats_path is None:
            datastats_path = os.path.join(self._cfg.checkpoint.ckpt_dir, f'data_stats.yaml')
        datastats_path = pathlib.Path(datastats_path)
        datastats_path.parent.mkdir(parents=False, exist_ok=True)
        datastats_to_save = dict_apply(datastats, lambda x: to_numpy(x).tolist())
        with open(datastats_path, 'w') as f:
            yaml.dump(datastats_to_save, f)
        return str(datastats_path.absolute())


    def save_checkpoint(self, 
                        tag='latest',
                        ckpt_path=None,  
                        exclude_keys=None, 
                        include_keys=None,
                        use_thread=True):
        '''
            save checkpoint
        '''
        if ckpt_path is None:
            ckpt_path = os.path.join(self._cfg.checkpoint.ckpt_dir, f'{tag}.ckpt')
        
        ckpt_path = pathlib.Path(ckpt_path)
        ckpt_path.parent.mkdir(parents=False, exist_ok=True)

        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)

        if include_keys is None:
            include_keys = tuple(self.include_keys)

        payload = {
            'cfg': self._cfg,
            'state_dicts': dict(),
            'pickles': dict()
        } 

        for key, value in self.__dict__.items():
            if hasattr(value, 'state_dict') and hasattr(value, 'load_state_dict'):
                # modules, optimizers and etc
                if key not in exclude_keys:
                    if use_thread:
                        payload['state_dicts'][key] = _copy_to_cpu(value.state_dict())
                    else:
                        payload['state_dicts'][key] = value.state_dict()
            
            elif key in include_keys:
                payload['pickles'][key] = dill.dumps(value)

        if use_thread:
            self._saving_thread = threading.Thread(
                target=lambda : torch.save(payload, ckpt_path.open('wb'), pickle_module=dill)
                )
            self._saving_thread.start()
        else:
            torch.save(payload, ckpt_path.open('wb'), pickle_module=dill)

        return str(ckpt_path.absolute())
    
    def get_checkpoint_path(self, tag='latest'):
        return pathlib.Path(self._ckpt_dir).joinpath('checkpoints', f'{tag}.ckpt')

    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = payload['pickles'].keys()

        for key, value in payload['state_dicts'].items():
            if key not in exclude_keys:
                self.__dict__[key].load_state_dict(value, **kwargs)

        for key in include_keys:
            if key in payload['pickles']:
                self.__dict__[key] = dill.loads(payload['pickles'][key])
    
    def load_checkpoint(self, 
                        path=None, 
                        tag='latest',
                        exclude_keys=None, 
                        include_keys=None, **kwargs):
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)

        payload = torch.load(path.open('rb'), pickle_module=dill, **kwargs)
        self.load_payload(payload, exclude_keys=exclude_keys, include_keys=include_keys)
        return payload
    
    @classmethod
    def create_from_checkpoint(cls, path, exclude_keys=None, 
                               include_keys=None, **kwargs):
        payload = torch.load(open(path, 'rb'), pickle_module=dill)
        instance = cls(payload['cfg'])
        instance.load_payload(
            payload=payload, 
            exclude_keys=exclude_keys,
            include_keys=include_keys,
            **kwargs)
        return instance

    def save_snapshot(self, tag='latest'):
        """
            Quick loading and saving for reserach, saves full state of the workspace.
        """
        path = pathlib.Path(self._ckpt_dir).joinpath(f'snapshot_{tag}.pkl')
        path.parent.mkdir(parents=False, exist_ok=True)
        torch.save(self, path.open('wb'), pickle_module=dill)
        return str(path.absolute())
    
    @classmethod
    def create_from_snapshot(cls, path):
        return torch.load(open(path, 'rb'), pickle_module=dill)