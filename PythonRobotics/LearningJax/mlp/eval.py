if __name__ == "__main__":
    import sys

    sys.path.append(
        "/Users/y.xtang/Documents/ML/beat_python/PythonRobotics/LearningJax"
    )


import os
import pathlib
import dill

import jax.numpy as jnp
from omegaconf import OmegaConf
from pprint import pprint
from model import MLP
from pprint import pprint


def eval():
    model_cfg = OmegaConf.load("./tmp/model_cfg.yaml")
    print(model_cfg)
    model = MLP(model_cfg.inp_dim, model_cfg.out_dim)

    with open("./tmp/checkpoints/epoch_90.pkl", "rb") as f:
        ckpt = dill.load(f)

    print(ckpt["pickles"])
    train_state = ckpt["train_state"]

    variables = {"params": train_state.params, "batch_stats": train_state.batch_stats}
    x = jnp.array(
        [
            [1.0],
            [
                3.0,
            ],
        ]
    )
    res = model.apply(variables, x, False)
    res2 = train_state.apply_fn(variables, x, False)
    print(res, res - res2)


def resume_train():
    from mlp_trainer import MLPTrainer

    trainer = MLPTrainer.create_from_checkpoint("./tmp/checkpoints/epoch_90.pkl")
    pprint(trainer.train_state)
    # trainer.run()


if __name__ == "__main__":
    # eval()
    resume_train()
