from diffusers import DDPMScheduler


if __name__ == "__main__":
    num_diffusion_iters = 100
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        beta_schedule="linear",
        clip_sample=True,
        prediction_type="epsilon",
    )
