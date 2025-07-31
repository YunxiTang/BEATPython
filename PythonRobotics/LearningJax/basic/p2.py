# import jax
# import flax.linen as nn
# from flax.training import train_state
# import optax
# from omegaconf import OmegaConf
# import copy
# from typing import Any
# import jax.numpy as jnp
# import numpy as np
# from torch.utils.data import Dataset, DataLoader, random_split


# def jnp_collate_fn(batch):
#     if isinstance(batch[0], np.ndarray):
#         return np.stack(batch)
#     elif isinstance(batch[0], (tuple,list)):
#         transposed = zip(*batch)
#         return [jnp_collate_fn(samples) for samples in transposed]
#     else:
#         return np.array(batch)


# class LinearRegDataset(Dataset):
#     def __init__(self, num_point: int = 5000) -> None:
#         super().__init__()
#         np.random.seed(0)
#         self._N = num_point
#         self._xs = np.random.normal(loc=0.0, scale=2.0, size=(self._N, 1))
#         noise = np.random.normal(loc=0.0, scale=0.1, size=(self._N, 1))
#         self._ys = self._xs * 5  - 1 + noise

#         self.data_size = num_point

#     def __len__(self):
#         return self._N

#     def __getitem__(self, index):

#         return self._xs[index], self._ys[index]


# class Linear(nn.Module):
#     @nn.compact
#     def __call__(self, x, train: bool):
#         x = nn.Dense(128, name='fc1')(x)
#         x = nn.Dropout(rate=0.1)(x, deterministic=not train)
#         x = nn.relu(x)
#         x = nn.BatchNorm(use_running_average=not train)(x)
#         x = nn.relu(x)
#         x = nn.Dense(1, name='fc2')(x)
#         return x


# class TrainState(train_state.TrainState):
#     batch_stats: Any = None
#     dropout_rng: Any = None


# class FlaxTrainer:
#     '''
#         A trainer class for flax model
#     '''
#     def __init__(self, cfg: OmegaConf, *inp_sample):
#         self.cfg = cfg

#         # create a rng_key for random streaming
#         self.rng_key = jax.random.PRNGKey(seed=self.cfg.train.seed)

#         # create a model
#         self.model: nn.Module
#         self.model = Linear()

#         # create an empty train state
#         self.train_state = None

#         # init the model & train state
#         self.train_state = self._init_train_state(*inp_sample)

#     def _init_train_state(self, *inp_args):
#         '''
#             - initialize the variables for the model,
#             - initialize the opt_state for the optimizer
#             - create a train state for the training
#         '''
#         # ============= model initialization ================
#         print('========= Model Initialization ==========')
#         params_key, dropout_key, self.rng_key = jax.random.split(self.rng_key, 3)
#         variables = self.model.init({'params': params_key, 'dropout': dropout_key}, *inp_args)
#         params = variables.get('params')
#         batch_stats = variables.get('batch_stats', {})
#         print('========= Model Initialization Done =========')

#         # ============= optimizer initialization =============
#         print('========= Optimizer Initialization =========')
#         optimizer = optax.adamw(learning_rate=self.cfg.train.lr)
#         opt_state = optimizer.init(params)
#         print('========= Optimizer Initialization Done =========')

#         # ============= assemble the train state =============
#         train_state = TrainState(
#             step=0,
#             apply_fn=self.model.apply,
#             params=params,
#             tx=optimizer,
#             opt_state=opt_state,
#             batch_stats=batch_stats,
#             dropout_rng=self.rng_key
#         )
#         return train_state


#     def create_function(self):
#         # loss function
#         def loss_fn(params, state: TrainState, batch, train: bool):
#             feats, labels = batch
#             model_variables = {'params': params, 'batch_stats': state.batch_stats}
#             output = state.apply_fn(model_variables,
#                                     feats, train=train,
#                                     rngs={'dropout': state.dropout_rng} if train else None,
#                                     mutable=['batch_stats'] if train else False)
#             if train:
#                 predicts, updated_model_state = output
#             else:
#                 predicts, updated_model_state = output, None
#             loss_val = jnp.mean( optax.l2_loss(predicts, labels) )
#             return loss_val, updated_model_state

#         # train step function
#         def train_step(state: TrainState, batch):
#             loss_val_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
#             (loss_value, updated_model_state), grads = loss_val_grad_fn(state.params, state, batch, train=True)
#             # update the dropout rng!
#             dropout_rng = jax.random.fold_in(state.dropout_rng, data=state.step)
#             updated_state = state.apply_gradients(
#                 grads=grads,
#                 batch_stats=updated_model_state['batch_stats'],
#                 dropout_rng=dropout_rng)
#             return {'loss': loss_value}, updated_state

#         # eval step function
#         def eval_step(state: TrainState, batch):
#             loss_value, _ = loss_fn( state.params, state, batch, train=False )
#             return {'loss': loss_value}

#         return train_step, eval_step


#     # def train_epoch(self, train_dataloader):

#     def train(self):
#         '''
#             train
#         '''
#         assert self.train_state is not None, 'Train state is None!'
#         cfg = copy.deepcopy(self.cfg)

#         # ========= configure the dataset ==========
#         dataset = LinearRegDataset()

#         # ========= configure the dataloader ========
#         data_size = dataset.data_size
#         train_size = int(0.8 * data_size)
#         val_size = data_size - train_size
#         print(f'Training Data Size: {train_size} Validation Datasize: {val_size}')
#         train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

#         train_dataloader = DataLoader(train_dataset,
#                                     batch_size=cfg.train.batch_size,
#                                     shuffle=True,
#                                     collate_fn=jnp_collate_fn,
#                                     drop_last=False )
#         val_dataloader = DataLoader(val_dataset,
#                                     batch_size=cfg.train.batch_size,
#                                     shuffle=False,
#                                     collate_fn=jnp_collate_fn,
#                                     drop_last=False )

#         # create the functions
#         train_step, eval_step = self.create_function()
#         jitted_train_step = jax.jit(train_step)
#         jitted_eval_step = jax.jit(eval_step)

#         # ========= main loop ===================
#         for epoch in range(cfg.train.max_epoch):
#             Loss = 0
#             for batch in train_dataloader:
#                 metric, self.train_state = jitted_train_step(self.train_state, batch)
#                 Loss += metric['loss']

#             if epoch % 20:
#                 eval_Loss = 0
#                 for batch in val_dataloader:
#                     metric = jitted_eval_step(self.train_state, batch)
#                     eval_Loss += metric['loss']

#                 print(f'Epoch: {epoch}: Train Loss: {Loss / train_size} || Eval Loss: {eval_Loss / val_size}')

#         return None

#     def save_checkpoint(self):
#         # TODO: add the checkpoint save logits
#         return None

# if __name__ == '__main__':

#     import time

#     cfg = OmegaConf.load( './cfg.yaml' )

#     dummy_sample = jax.random.normal(jax.random.PRNGKey(0), shape=[2, 1])

#     inp_args = [dummy_sample, False]
#     trainer = FlaxTrainer(cfg, *inp_args)

#     tc = time.time()

#     trainer.train()

#     print('elapsed time: ', time.time() - tc)

import jax
import flax.linen as nn
from flax.training import train_state
import optax
from omegaconf import OmegaConf
import copy
from typing import Any
import jax.numpy as jnp
import numpy as np
from datasets import Dataset, DatasetDict
import time


def jnp_collate_fn(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [jnp_collate_fn(samples) for samples in transposed]
    else:
        return np.array(batch)


def create_data(num_point: int = 5000):
    np.random.seed(0)
    xs = np.random.normal(loc=0.0, scale=2.0, size=(num_point, 1))
    noise = np.random.normal(loc=0.0, scale=0.1, size=(num_point, 1))
    ys = xs * 5 - 1 + noise
    return xs, ys


class Linear(nn.Module):
    @nn.compact
    def __call__(self, x, train: bool):
        x = nn.Dense(1280, name="fc1")(x)
        x = nn.Dropout(rate=0.1)(x, deterministic=not train)
        x = nn.relu(x)
        x = nn.Dense(1280 * 2, name="fc2")(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        x = nn.Dense(1, name="fc3")(x)
        return x


class TrainState(train_state.TrainState):
    batch_stats: Any = None
    dropout_rng: Any = None


class FlaxTrainer:
    """
    A trainer class for flax model
    """

    def __init__(self, cfg: OmegaConf, inp_sample):
        self.cfg = cfg

        # create a rng_key for random streaming
        self.rng_key = jax.random.PRNGKey(seed=self.cfg.train.seed)

        # create a model
        self.model: nn.Module
        self.model = Linear()

        # create an empty train state
        self.train_state = None

        # init the model & train state
        self.train_state = self._init_train_state(inp_sample)

    def _init_train_state(self, inp_sample):
        """
        - initialize the variables for the model,
        - initialize the opt_state for the optimizer
        - create a train state for the training
        """
        # ============= model initialization ================
        print("========= Model Initialization ==========")
        params_key, dropout_key, self.rng_key = jax.random.split(self.rng_key, 3)
        variables = self.model.init(
            {"params": params_key, "dropout": dropout_key}, inp_sample, False
        )
        params = variables.get("params")
        batch_stats = variables.get("batch_stats", {})
        print("========= Model Initialization Done =========")

        # ============= optimizer initialization =============
        print("========= Optimizer Initialization =========")
        optimizer = optax.adamw(learning_rate=self.cfg.train.lr)
        opt_state = optimizer.init(params)
        print("========= Optimizer Initialization Done =========")

        # ============= assemble the train state =============
        train_state = TrainState(
            step=0,
            apply_fn=self.model.apply,
            params=params,
            tx=optimizer,
            opt_state=opt_state,
            batch_stats=batch_stats,
            dropout_rng=self.rng_key,
        )
        return train_state

    def create_function(self):
        # loss function
        def loss_fn(params, state: TrainState, batch, train: bool):
            feats = batch["x"]
            labels = batch["y"]
            model_variables = {"params": params, "batch_stats": state.batch_stats}
            output = state.apply_fn(
                model_variables,
                feats,
                train=train,
                rngs={"dropout": state.dropout_rng} if train else None,
                mutable=["batch_stats"] if train else False,
            )
            if train:
                predicts, updated_model_state = output
            else:
                predicts, updated_model_state = output, None
            loss_val = jnp.mean(optax.l2_loss(predicts, labels))
            return loss_val, updated_model_state

        # train step function
        def train_step(state: TrainState, batch):
            loss_val_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss_value, updated_model_state), grads = loss_val_grad_fn(
                state.params, state, batch, train=True
            )
            # update the dropout rng!
            dropout_rng = jax.random.fold_in(state.dropout_rng, data=state.step)
            updated_state = state.apply_gradients(
                grads=grads,
                batch_stats=updated_model_state["batch_stats"],
                dropout_rng=dropout_rng,
            )
            return {"loss": loss_value}, updated_state

        # eval step function
        def eval_step(state: TrainState, batch):
            loss_value, _ = loss_fn(state.params, state, batch, train=False)
            return {"loss": loss_value}

        return train_step, eval_step

    # def train_epoch(self, train_dataloader):

    def train(self):
        """
        train
        """
        assert self.train_state is not None, "Train state is None!"
        cfg = copy.deepcopy(self.cfg)

        # ========= configure the dataset ==========
        num_point = 50000
        xs, ys = create_data(num_point)
        dataset = Dataset.from_dict({"x": xs, "y": ys}).with_format(
            "jax", device=str(jax.devices("gpu")[0])
        )

        # ========= configure the dataloader ========

        # create the functions
        train_step, eval_step = self.create_function()
        jitted_train_step = jax.jit(train_step)
        jitted_eval_step = jax.jit(eval_step)

        # ========= main loop ===================

        for epoch in range(cfg.train.max_epoch):
            Loss = 0
            tc = time.time()
            for batch in dataset.iter(cfg.train.batch_size):
                metric, self.train_state = jitted_train_step(self.train_state, batch)
                Loss += metric["loss"]
            print("elapsed time: ", time.time() - tc)

            if epoch % 20:
                print(f"Epoch: {epoch}: Train Loss: {Loss}")
        return None

    def save_checkpoint(self):
        # TODO: add the checkpoint save logits
        return None


if __name__ == "__main__":
    cfg = OmegaConf.load("./cfg.yaml")

    dummy_sample = jax.random.normal(jax.random.PRNGKey(0), shape=[2, 1])
    dummy_sample = jax.device_put(dummy_sample, jax.devices()[0])
    trainer = FlaxTrainer(cfg, dummy_sample)

    trainer.train()
