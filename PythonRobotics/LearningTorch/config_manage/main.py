from omegaconf import DictConfig, OmegaConf
from pathlib import Path


def main(cfg_path):
    cfg = OmegaConf.load(cfg_path)
    # To access elements of the config
    print(f"The batch size is {cfg.batch_size}")
    print(f"The learning rate is {cfg['lr']}")
    return cfg

if __name__ =='__main__':
    main(Path('PythonRobotics/LearningTorch/config_manage/config/train_cfg.yaml'))