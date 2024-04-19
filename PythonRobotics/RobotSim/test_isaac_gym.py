from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym import terrain_utils
from isaacgym import torch_utils
import torch
import numpy as np
import random
import pathlib, os


if __name__ == '__main__':
    gym = gymapi.acquire_gym()
    args = gymutil.parse_arguments(description="Isaac Gym Example")
    
    # ============= create a simulation =======================
    # get default set of parameters
    sim_params = gymapi.SimParams()

    # set common parameters
    sim_params.dt = 1 / 60
    sim_params.substeps = 2
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

    # set PhysX-specific parameters
    sim_params.physx.use_gpu = True
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 6
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.contact_offset = 0.01
    sim_params.physx.rest_offset = 0.0

    sim_params.use_gpu_pipeline = args.use_gpu_pipeline
    device = args.sim_device if args.use_gpu_pipeline else 'cpu'
    
    sim = gym.create_sim(args.compute_device_id, 
                         args.compute_device_id, 
                         gymapi.SIM_PHYSX, 
                         sim_params)
    if sim is None:
        print("*** Failed to create sim")
        quit()

    # creating a ground plane
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
    plane_params.distance = 0
    plane_params.static_friction = 1
    plane_params.dynamic_friction = 1
    plane_params.restitution = 0.5

    gym.add_ground(sim, plane_params)

    # loading assets
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = False
    asset_options.armature = 0.01
    asset_root = os.path.join(os.path.dirname(os.path.realpath(__file__)),  'assets')
    asset_file = "simple_model/cube.urdf"
    asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

    num_bodies = gym.get_asset_rigid_body_count(asset)

    num_envs = 16
    envs_per_row = 8
    env_spacing = 2.0
    env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
    env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

    # cache some common handles for later use
    envs = []
    actor_handles = []

    # create and populate the environments
    for i in range(num_envs):
        env = gym.create_env(sim, 
                             env_lower,
                             env_upper, 
                             envs_per_row)
        envs.append(env)

        c = 0.5 + 0.5 * np.random.random(3)
        color = gymapi.Vec3(c[0], c[1], c[2])

        height = random.uniform(1.0, 2.5)

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, height)

        actor_handle = gym.create_actor(env, asset, pose, f"MyActor_{i}", i, 1)
        actor_handles.append(actor_handle)
        gym.set_rigid_body_color(env, actor_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

    cam_props = gymapi.CameraProperties()
    viewer = gym.create_viewer(sim, cam_props)

    gym.prepare_sim(sim)

    torque_amt = 15

    frame_count = 0

    while not gym.query_viewer_has_closed(viewer):

        if (frame_count - 99) % 200 == 0:
            # set forces and torques for the ant root bodies
            forces = torch.zeros((num_envs, num_bodies, 3), 
                                 device=device, 
                                 dtype=torch.float)
            torques = torch.zeros((num_envs, num_bodies, 3), 
                                  device=device, 
                                  dtype=torch.float)
            forces[:, 0, 1] = 0.0
            torques[:, 0, 2] = torque_amt

            gym.apply_rigid_body_force_tensors(sim, 
                                               gymtorch.unwrap_tensor(forces), 
                                               gymtorch.unwrap_tensor(torques), 
                                               gymapi.ENV_SPACE)
            torque_amt = -torque_amt
        
        
        # step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        # update the viewer
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

        frame_count += 1

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
