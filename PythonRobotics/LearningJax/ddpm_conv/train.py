if __name__ == "__main__":
    import sys

    sys.path.append(
        "/Users/y.xtang/Documents/ML/beat_python/PythonRobotics/LearningJax"
    )


from omegaconf import OmegaConf
import jax
import os
from datasets import load_dataset
from scheduler import DDPMScheduler
from ddpm_trainer import DDPMTrainer


if __name__ == "__main__":
    ds = load_dataset("ylecun/mnist")
    ds.set_format("jax", device=str(jax.devices()[0]))
    train_ds = ds["train"].shuffle(12)
    test_ds = ds["test"]

    workspace_dir = os.path.dirname(os.path.realpath(__file__))
    cfg_path = os.path.join(workspace_dir, "train_ddpm_config.yaml")
    cfg = OmegaConf.load(cfg_path)
    trainer = DDPMTrainer(cfg)

    scheduler = DDPMScheduler(timesteps=1000, seed=0)
