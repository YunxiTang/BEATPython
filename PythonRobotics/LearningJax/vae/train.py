import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from typing import Any
from flax.training import train_state
import optax
from datasets import Dataset, load_dataset


@jax.vmap
def kl_divergence(mean, logvar):
    return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))


@jax.vmap
def binary_cross_entropy_with_logits(logits, labels):
    logits = nn.log_sigmoid(logits)
    return jnp.sum(optax.sigmoid_binary_cross_entropy(logits, labels))


# @jax.jit
def loss_fn(logits, labels, mean, logvar):
    kl_loss = kl_divergence(mean, logvar)
    bce_loss = binary_cross_entropy_with_logits(logits, labels)
    loss = kl_loss + bce_loss
    return loss, kl_loss, bce_loss


class Encoder(nn.Module):
    latent_dim: int = 2

    def setup(self):
        self.fc1 = nn.Dense(512)
        self.fc_mean = nn.Dense(self.latent_dim)
        self.fc_logvar = nn.Dense(
            self.latent_dim,
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.zeros_init(),
        )

    def __call__(self, x):
        x = self.fc1(x)
        x = nn.gelu(x)
        mean_x = self.fc_mean(x)
        logvar_x = self.fc_logvar(x)
        return mean_x, logvar_x


class Decoder(nn.Module):
    output_dim: int

    def setup(self):
        self.fc1 = nn.Dense(512)
        self.fc_cond = nn.Dense(512)
        self.fc2 = nn.Dense(self.output_dim)

    def __call__(self, z, label):
        z = self.fc1(z)
        z = nn.gelu(z)

        cond = self.fc_cond(label)
        cond = nn.gelu(cond)

        z = z + cond
        z = self.fc2(z)
        return z


class VAE(nn.Module):
    latent_dim: int
    output_dim: int

    def setup(self):
        self.encoder = Encoder(self.latent_dim)
        self.decoder = Decoder(self.output_dim)

    @staticmethod
    def reparameterize(mean, logvar, rng):
        std = jnp.exp(0.5 * logvar)
        eps = random.normal(rng, logvar.shape)
        return mean + eps * std

    def __call__(self, x, label, train: bool):
        # encoder
        z_mean, z_logvar = self.encoder(x)
        rng = self.make_rng("latent_sample")
        # differentiable reparameterization
        z_sample = self.reparameterize(z_mean, z_logvar, rng)
        # decode
        res = self.decoder(z_sample, label)
        return res, z_mean, z_logvar

    def generate(self, z, label):
        return nn.sigmoid(self.decoder(z, label))


class TrainState(train_state.TrainState):
    batch_stats: Any = None
    dropout_rng: Any = None


class FlaxTrainer:
    """
    A trainer class for flax model
    """

    def __init__(self, *inp_sample):
        # create a rng_key for random streaming
        self.rng_key = jax.random.PRNGKey(seed=10)

        # create a model
        self.model: nn.Module
        self.model = VAE(2, 784)

        # create an empty train state
        self.train_state = None

        # init the model & train state
        self.train_state = self._init_train_state(*inp_sample)

    def _init_train_state(self, *inp_sample):
        """
        - initialize the variables for the model,
        - initialize the opt_state for the optimizer
        - create a train state for the training
        """
        # ============= model initialization ================
        print("========= Model Initialization ==========")
        params_key, dropout_key, self.rng_key = jax.random.split(self.rng_key, 3)
        variables = self.model.init(
            {"params": params_key, "latent_sample": dropout_key}, *inp_sample
        )
        params = variables.get("params")
        batch_stats = variables.get("batch_stats", {})
        print("========= Model Initialization Done =========")

        # ============= optimizer initialization =============
        print("========= Optimizer Initialization =========")
        optimizer = optax.adamw(learning_rate=0.001)
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
            feats = batch["image"].reshape(-1, 28 * 28)
            labels = batch["label"]
            labels = nn.one_hot(labels, num_classes=10)
            model_variables = {"params": params, "batch_stats": state.batch_stats}
            output = state.apply_fn(
                model_variables,
                feats,
                labels,
                train,
                rngs={"latent_sample": state.dropout_rng} if train else None,
                mutable=["batch_stats"] if train else False,
            )
            if train:
                predicts, updated_model_state = output
            else:
                predicts, updated_model_state = output, None
            res, z_mean, z_logvar = predicts
            loss_val = (
                1.01 * kl_divergence(z_mean, z_logvar)
                + 0.0 * binary_cross_entropy_with_logits(res, feats)
                + jnp.mean(optax.huber_loss(res, feats), axis=1)
            )
            return jnp.mean(loss_val), updated_model_state

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

    def train(self):
        """
        train
        """
        assert self.train_state is not None, "Train state is None!"

        # ========= configure the dataset ==========
        ds = load_dataset(
            "ylecun/mnist", cache_dir="/home/yxtang/CodeBase/PythonCourse/dataset"
        )
        ds.set_format("jax", device=str(jax.devices()[0]))
        train_ds = ds["train"].shuffle(12)
        test_ds = ds["test"]
        train_dataset = train_ds
        # ========= configure the dataloader ========

        # create the functions
        train_step, eval_step = self.create_function()
        jitted_train_step = jax.jit(train_step)
        jitted_eval_step = jax.jit(eval_step)

        num = 10
        # latent_vars = jax.random.normal(random.PRNGKey(256), [num, 2]) * 10
        # latent_means = jax.random.randint(random.PRNGKey(26), [num, 2], minval=-5, maxval=5)
        # latents = latent_means + latent_vars
        labels = nn.one_hot(jnp.arange(0, num), num_classes=10)
        print(labels)
        latents1 = jax.random.normal(random.PRNGKey(2506), [num, 2])
        latents2 = jax.random.normal(random.PRNGKey(25060), [num, 2])
        # ========= main loop ===================
        for epoch in range(100):
            Loss = 0
            for batch in train_dataset.iter(512):
                metric, self.train_state = jitted_train_step(self.train_state, batch)
                Loss += metric["loss"]
            print(f"Epoch: {epoch}: Train Loss: {Loss}")

            if epoch % 5 == 0:
                variables = {
                    "params": self.train_state.params,
                    "batch_stats": self.train_state.batch_stats,
                }

                res1 = model.apply(variables, latents1, labels, method=VAE.generate)
                res2 = model.apply(variables, latents2, labels, method=VAE.generate)
                res1 = res1.reshape(-1, 28, 28)
                imgs1 = [np.array(res1[i]) for i in range(num)]
                img1 = np.concatenate(imgs1, axis=1)

                res2 = res2.reshape(-1, 28, 28)
                imgs2 = [np.array(res2[i]) for i in range(num)]
                img2 = np.concatenate(imgs2, axis=1)

                img = np.concatenate([img1, img2], axis=0)
                plt.imsave(f"res/epoch_{epoch}.png", img, cmap=cm.gray)
        return None


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    ds = load_dataset(
        "ylecun/mnist", cache_dir="/home/yxtang/CodeBase/PythonCourse/dataset"
    )
    ds.set_format("jax")
    train_ds = ds["train"]
    test_ds = ds["test"]
    sample_label = train_ds[0:2]["label"]
    sample_label = nn.one_hot(sample_label, num_classes=10)

    sample_img = train_ds[0:2]["image"]

    sample_img = sample_img.reshape(-1, 28 * 28)
    model = VAE(2, 784)
    root_rng = random.PRNGKey(100)
    model_init_rng, reparam_rng = random.split(root_rng, 2)
    variables = model.init(
        {"params": model_init_rng, "latent_sample": reparam_rng},
        sample_img,
        sample_label,
        False,
    )

    trainer = FlaxTrainer(sample_img, sample_label, False)
    trainer.train()
