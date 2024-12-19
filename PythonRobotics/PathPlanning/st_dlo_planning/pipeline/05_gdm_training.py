'''
    train a global deformation model
'''
import os
from omegaconf import OmegaConf
from st_dlo_planning.neural_mpc_tracker.gdm_trainer import GDMTrainer


if __name__ == '__main__':
    cfg_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 
                           'gdm_train_B.yaml')

    cfg = OmegaConf.load(cfg_path)
    trainer = GDMTrainer(cfg)
    trainer.run()
