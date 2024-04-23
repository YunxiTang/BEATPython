from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch

import torch
import math
import numpy as np

from isaacgym.torch_utils import quat_conjugate, quat_mul

import os


def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


if __name__ == '__main__':
    
    args = gymutil.parse_arguments(description="Isaac Gym Operational Space Control")

    
    # initialize gym
    gym = gymapi.acquire_gym()

    # config the sim parameters
    sim_params = gymapi.SimParams()

    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.dt = 1.0 / 60.0

    sim_params.substeps = 2
    # set PhysX-specific parameters
    sim_params.physx.use_gpu = True
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 6
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.contact_offset = 0.01
    sim_params.physx.rest_offset = 0.0

    # create a simulation
    sim = gym.create_sim(args.compute_device_id, 
                         args.graphics_device_id, 
                         args.physics_engine, 
                         sim_params)
    
    # create viewer
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())

    # add ground plane
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane_params)

    # Load franka asset
    asset_root = os.path.join(os.path.dirname(os.path.realpath(__file__)),  'assets')
    franka_asset_file = "urdf/franka_description/robots/franka_panda.urdf"

    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.flip_visual_attachments = True
    asset_options.armature = 0.01
    asset_options.disable_gravity = False #True # 

    asset = gym.load_asset(sim, asset_root, franka_asset_file, asset_options)

    num_bodies = gym.get_asset_rigid_body_count(asset)
    print( num_bodies )
    quit()

    # create sub env
    num_envs = 16
    envs_per_row = 8
    env_spacing = 2.0
    env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
    env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

    envs = []
    actor_handles = []
    for i in range(num_envs):
        env = gym.create_env(sim, 
                             env_lower,
                             env_upper, 
                             envs_per_row)
        envs.append(env)
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, 0.0)

        actor_handle = gym.create_actor(env, asset, pose, f"MyActor_{i}", i, 1)
        actor_handles.append(actor_handle)

    gym.prepare_sim(sim)

    while not gym.query_viewer_has_closed(viewer):
        
        # acquire tensor descriptors
        # root_states_desc = gym.acquire_actor_root_state_tensor(sim)
        # root_states = gymtorch.wrap_tensor(root_states_desc)
        # root_states_vec = root_states.view(num_envs, num_actor_per_env, 13)
        # print('pos', root_states_vec[0,0,0:3])

        # step the physics
        gym.simulate(sim)

        gym.refresh_actor_root_state_tensor(sim)
        gym.refresh_dof_state_tensor(sim)
        gym.fetch_results(sim, True)

        # update the viewer
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
    