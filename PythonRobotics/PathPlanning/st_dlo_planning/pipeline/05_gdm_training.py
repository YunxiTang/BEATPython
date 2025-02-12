if __name__ == '__main__':
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)
    
    import os

from omegaconf import OmegaConf
from st_dlo_planning.neural_mpc_tracker.gdm_trainer import GDMTrainer
from st_dlo_planning.neural_mpc_tracker.configuration_gdm import GDM_CFG


if __name__ == '__main__':
    cfg_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 
                           'gdm_train_B.yaml')
    gdm_cfg = GDM_CFG()
    train_cfg = OmegaConf.load(cfg_path)
    
    trainer = GDMTrainer(train_cfg, gdm_cfg)
    trainer.run()