"""
Configuration settings and constants for 3D Gaussian Splatting with NeRF datasets.
"""
import warp as wp
import numpy as np
import random

SEED = 42
random.seed(SEED)

# Warp data types and constants (keep capitalized as they are types)
WP_FLOAT16 = wp.float16
WP_FLOAT32 = wp.float32
WP_INT = wp.int32
WP_VEC2 = wp.vec2
WP_VEC2H = wp.vec2h
VEC6 = wp.types.vector(length=6, dtype=WP_FLOAT32)
DEVICE = "cuda:0" # Use "cpu" or "cuda"

TILE_M = wp.constant(16)
TILE_N = wp.constant(16)
TILE_THREADS = wp.constant(256)


class GaussianParams:
    """Parameters for 3D Gaussian Splatting."""

    # Training parameters
    num_iterations = 20000  # Default number of training iterations
    num_points = 5000 # Initial number of Gaussian points

    # Simple learning rate scheduler configuration
    use_lr_scheduler = True
    lr_scheduler_config = {
        'lr_pos': 1e-3,      # Initial learning rate for positions
        'lr_scale': 5e-3,    # Initial learning rate for scales  
        'lr_rot': 5e-3,      # Initial learning rate for rotations
        'lr_sh': 2e-3,       # Initial learning rate for spherical harmonics
        'lr_opac': 5e-3,     # Initial learning rate for opacities
        'final_lr_factor': 0.01  # Final LR will be 1% of initial LR
    }

    # Optimization parameters
    densification_interval = 100  # Perform densification every N iterations
    pruning_interval = 100  # Perform pruning every N iterations
    opacity_reset_interval = 3000
    save_interval = 300  # Save checkpoint every N iterations
    adam_beta1 = 0.9  # Adam optimizer beta1 parameter
    adam_beta2 = 0.999  # Adam optimizer beta2 parameter
    adam_epsilon = 1e-8  # Adam optimizer epsilon parameter
    
    
    densify_grad_threshold = 0.0002
    cull_opacity_threshold = 0.005
    start_prune_iter = 500
    end_prune_iter = 15000
    percent_dense = 0.01
    max_allowed_prune_ratio = 1.0 # no limit on pruning ratio

    # Gaussian parameters
    initial_scale = 0.1  # Initial scale for Gaussian points
    scale_modifier = 1.0  # Scaling factor for Gaussian splats
    sh_degree = 3  # Spherical harmonics degree

    # Scene parameters
    scene_scale = 1.0  # Scale factor for the scene
    background_color = [0.0, 0.0, 0.0]  # White background for NeRF synthetic

    # Loss parameters
    lambda_dssim = 0.0  # Weight for SSIM loss (1.0 means only SSIM, 0.0 means only L1)
    
    # Depth loss parameters
    depth_l1_weight_init = 0.0  # Initial weight for depth L1 loss
    depth_l1_weight_final = 0.0  # Final weight for depth L1 loss
    depth_l1_delay_steps = 0  # Number of steps to delay depth loss
    depth_l1_delay_mult = 0.0  # Multiplier for delay rate
    
    near = 0.01  # Default near clipping plane
    far = 100.0  # Default far clipping plane

  
    @classmethod
    def get_depth_l1_weight(cls, step):
        """Compute the depth L1 loss weight for the current step.
        
        Args:
            step (int): Current training step
            
        Returns:
            float: Weight for depth L1 loss
        """
        if step < 0 or (cls.depth_l1_weight_init == 0.0 and cls.depth_l1_weight_final == 0.0):
            # Disable depth loss
            return 0.0
            
        if cls.depth_l1_delay_steps > 0:
            # A kind of reverse cosine decay
            delay_rate = cls.depth_l1_delay_mult + (1 - cls.depth_l1_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / cls.depth_l1_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
            
        # Logarithmic interpolation between initial and final weights
        t = np.clip(step / cls.num_iterations, 0, 1)
        log_lerp = np.exp(np.log(cls.depth_l1_weight_init) * (1 - t) + np.log(cls.depth_l1_weight_final) * t)
        
        return delay_rate * log_lerp

    @classmethod
    def update(cls, **kwargs):
        """Update parameters with new values."""
        for key, value in kwargs.items():
            if hasattr(cls, key):
                setattr(cls, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key}")

    @classmethod
    def get_config_dict(cls):
        """Get parameters as a dictionary."""
        return {
            'num_iterations': cls.num_iterations,
            'num_points': cls.num_points,
            'densification_interval': cls.densification_interval,
            'pruning_interval': cls.pruning_interval,
            'scale_modifier': cls.scale_modifier,
            'sh_degree': cls.sh_degree,
            'background_color': cls.background_color,
            'save_interval': cls.save_interval,
            'adam_beta1': cls.adam_beta1,
            'adam_beta2': cls.adam_beta2,
            'adam_epsilon': cls.adam_epsilon,
            'initial_scale': cls.initial_scale,
            'scene_scale': cls.scene_scale,
            'near': cls.near,
            'far': cls.far,
            'lambda_dssim': cls.lambda_dssim,
            'depth_l1_weight_init': cls.depth_l1_weight_init,
            'depth_l1_weight_final': cls.depth_l1_weight_final,
            'depth_l1_delay_steps': cls.depth_l1_delay_steps,
            'depth_l1_delay_mult': cls.depth_l1_delay_mult,
            'densify_grad_threshold': cls.densify_grad_threshold,
            'cull_opacity_threshold': cls.cull_opacity_threshold,
            'start_prune_iter': cls.start_prune_iter,
            'end_prune_iter': cls.end_prune_iter,
            'use_lr_scheduler': cls.use_lr_scheduler,
            'lr_scheduler_config': cls.lr_scheduler_config,
            'max_allowed_prune_ratio': cls.max_allowed_prune_ratio,
        }
