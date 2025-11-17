import jax


class EMA:
    """
    Exponential Moving Average (EMA) for Flax Model
    """

    def __init__(self, model_params, decay_rate: float):
        self.decay_rate = decay_rate
        self.ema_params = model_params

    def init(self, model_params):
        self.ema_params = model_params

    def update(self, model_params):
        if self.ema_params is None:
            raise ValueError("EMA parameters must be initialized before updating.")

        self.ema_params = jax.tree_util.tree_map(
            lambda ema_param, param: self.decay_rate * ema_param
            + (1.0 - self.decay_rate) * param,
            self.ema_params,
            model_params,
        )

    def get_ema_params(self):
        return self.ema_params


if __name__ == "__main__":
    import torch.nn as nn
