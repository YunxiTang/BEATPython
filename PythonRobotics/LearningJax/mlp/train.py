if __name__ == '__main__':
    import sys
    sys.path.append('/Users/y.xtang/Documents/ML/beat_python/PythonRobotics/LearningJax')


import os
import pathlib
from omegaconf import OmegaConf
from pprint import pprint
from mlp_trainer import MLPTrainer


if __name__ == '__main__':
    workspace_dir = os.path.dirname(os.path.realpath(__file__))
    cfg_path = os.path.join(workspace_dir, 'train_mlp_config.yaml')
    cfg = OmegaConf.load(cfg_path)
    # pprint(cfg)
    trainer = MLPTrainer(cfg)
    
    trainer.run()
    
    