import jax.random as random
import flax.linen as nn
import jax

from datasets import load_dataset
from scheduler import DDPMScheduler
from utils import FlaxTrainer


if __name__ == "__main__":
    from unet2d import CondUnet2D

    ds = load_dataset(
        "ylecun/mnist", cache_dir="/home/yxtang/CodeBase/PythonCourse/dataset"
    )
    ds.set_format("jax", device=str(jax.devices()[0]))
    train_ds = ds["train"].shuffle(12)
    test_ds = ds["test"]

    scheduler = DDPMScheduler(timesteps=1000, seed=0)

    num = 10
    sample_img = train_ds[0 : 0 + num]["image"][..., None] / 127.5 - 1
    sample_label = train_ds[0 : 0 + num]["label"]

    # sample noise to add to data points
    noises = random.normal(random.key(0), shape=sample_img.shape)
    timesteps = random.randint(
        random.key(0),
        shape=[
            sample_img.shape[0],
        ],
        minval=0,
        maxval=scheduler.timesteps,
    )
    noisy_sample = scheduler.add_noise(
        sample_img, noises, timesteps
    )  # forward diffusion process
    label_conds = nn.one_hot(sample_label, num_classes=10)

    model = CondUnet2D(
        64,
        64,
        in_channel=1,
        kernel_size=(5, 5),
        basic_channel=16,
        channel_scale_factor=(4, 8),
        num_groups=8,
    )

    trainer = FlaxTrainer(model, scheduler, noisy_sample, timesteps, label_conds, False)
    trainer.train(train_ds)
