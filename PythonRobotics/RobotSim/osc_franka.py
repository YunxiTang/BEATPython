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
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline
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

    # create table asset
    table_dims = gymapi.Vec3(1.0, 0.5, 0.4)
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, asset_options)

    franka_asset_file = "urdf/franka_description/robots/franka_panda.urdf"
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.flip_visual_attachments = True
    asset_options.armature = 0.01
    asset_options.disable_gravity = True # False #
    franka_asset = gym.load_asset(sim, asset_root, franka_asset_file, asset_options)

    mug_asset_file = 'urdf/ycb/025_mug/025_mug.urdf'
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = False
    asset_options.flip_visual_attachments = True
    asset_options.armature = 0.01
    asset_options.disable_gravity = False #
    mug_asset = gym.load_asset(sim, asset_root, mug_asset_file, asset_options)
    
    # franka info
    franka_num_bodies = gym.get_asset_rigid_body_count(franka_asset)
    franka_dof_props = gym.get_asset_dof_properties(franka_asset)
    franka_lower_limits = franka_dof_props['lower']
    franka_upper_limits = franka_dof_props['upper']
    franka_ranges = franka_upper_limits - franka_lower_limits
    franka_mids = 0.5 * (franka_upper_limits + franka_lower_limits)
    franka_num_dofs = len(franka_dof_props)

    # set default DOF states
    default_dof_state = np.zeros(franka_num_dofs, dtype=gymapi.DofState.dtype)
    default_dof_state["pos"][:7] = franka_mids[:7]

    # set DOF control properties (except grippers)
    franka_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_EFFORT)
    franka_dof_props["stiffness"][:7].fill(0.0)
    franka_dof_props["damping"][:7].fill(0.0)

    # set DOF control properties for grippers
    franka_dof_props["driveMode"][7:].fill(gymapi.DOF_MODE_POS)
    franka_dof_props["stiffness"][7:].fill(800.0)
    franka_dof_props["damping"][7:].fill(40.0)

    # create sub-env
    num_envs = 25
    envs_per_row = 5
    env_spacing = 1.50
    env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
    env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

    envs = []
    actor_handles = []
    init_pos_list = []
    init_orn_list = []
    hand_idxs = []
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.0, 0.0)

    table_pose = gymapi.Transform()
    table_pose.p = gymapi.Vec3(0.0, table_dims.y, 0.0)

    mug_pose = gymapi.Transform()
    mug_pose.p = gymapi.Vec3(0.0, table_dims.y, 1.0)
    
    for i in range(num_envs):
        env = gym.create_env(sim, 
                             env_lower,
                             env_upper, 
                             envs_per_row)
        envs.append(env)

        franka_handle = gym.create_actor(env, franka_asset, pose, f"franka", i, 0)
        table_handle = gym.create_actor(env, table_asset, table_pose, f"table", i, 0)

        mug_pose.r = gymapi.Quat.from_euler_zyx(math.pi/2+np.random.random(1)[0], 0, math.pi/2+np.random.random(1)[0])
        mug_handle = gym.create_actor(env, mug_asset, mug_pose, f"table", i, 0)

        # Set initial DOF states
        gym.set_actor_dof_states(env, franka_handle, default_dof_state, gymapi.STATE_ALL)
        gym.set_actor_dof_properties(env, franka_handle, franka_dof_props)

        # Get inital hand pose
        hand_handle = gym.find_actor_rigid_body_handle(env, franka_handle, "panda_hand")
        hand_pose = gym.get_rigid_transform(env, hand_handle)

        init_pos_list.append([hand_pose.p.x, hand_pose.p.y, hand_pose.p.z])
        init_orn_list.append([hand_pose.r.x, hand_pose.r.y, hand_pose.r.z, hand_pose.r.w])

        # Get global index of hand in rigid body state tensor
        hand_idx = gym.find_actor_rigid_body_index(env, franka_handle, "panda_hand", gymapi.DOMAIN_SIM)
        hand_idxs.append(hand_idx)

        # color setting
        c = 0.5 + 0.5 * np.random.random(3)
        color = gymapi.Vec3(c[0], c[1], c[2])
        gym.set_rigid_body_color(env, table_handle, 0, gymapi.MESH_VISUAL, color)

        for k in range(franka_num_bodies):
            c = 0.5 + 0.5 * np.random.random(3)
            color = gymapi.Vec3(c[0], c[1], c[2])
            gym.set_rigid_body_color(env, franka_handle, k, gymapi.MESH_VISUAL, color)
        actor_handles.append(franka_handle)

    gym.prepare_sim(sim)
    # initial hand position and orientation tensors
    init_pos = torch.Tensor(init_pos_list).view(num_envs, 3)
    init_orn = torch.Tensor(init_orn_list).view(num_envs, 4)
    if args.use_gpu_pipeline:
        init_pos = init_pos.to('cuda:0')
        init_orn = init_orn.to('cuda:0')

    # desired hand positions and orientations
    pos_des = init_pos.clone()
    orn_des = init_orn.clone()

    # Prepare jacobian tensor
    # For franka, tensor shape is (num_envs, 10, 6, 9)
    _jacobian = gym.acquire_jacobian_tensor(sim, "franka")
    jacobian = gymtorch.wrap_tensor(_jacobian)

    # Jacobian entries for end effector
    hand_index = gym.get_asset_rigid_body_dict(franka_asset)["panda_hand"]
    j_eef = jacobian[:, hand_index - 1, :]

    # Prepare mass matrix tensor
    # For franka, tensor shape is (num_envs, 9, 9)
    _massmatrix = gym.acquire_mass_matrix_tensor(sim, "franka")
    mm = gymtorch.wrap_tensor(_massmatrix)

    kp = 5
    kv = 2 * math.sqrt(kp) 

    # Rigid body state tensor
    _rb_states = gym.acquire_rigid_body_state_tensor(sim)
    rb_states = gymtorch.wrap_tensor(_rb_states)

    # DOF state tensor
    _dof_states = gym.acquire_dof_state_tensor(sim)
    dof_states = gymtorch.wrap_tensor(_dof_states)
    dof_vel = dof_states[:, 1].view(num_envs, 9, 1)
    dof_pos = dof_states[:, 0].view(num_envs, 9, 1)
    itr = 0

    num_actor_per_env = 1
    while not gym.query_viewer_has_closed(viewer):
        # Randomize desired hand orientations
        if itr % 20 == 0:
            orn_des = torch.rand_like(orn_des)
            orn_des /= torch.norm(orn_des)

        itr += 1

        # Update jacobian and mass matrix
        gym.refresh_rigid_body_state_tensor(sim)
        gym.refresh_dof_state_tensor(sim)
        gym.refresh_jacobian_tensors(sim)
        gym.refresh_mass_matrix_tensors(sim)

        # Get current hand poses
        pos_cur = rb_states[hand_idxs, :3]
        orn_cur = rb_states[hand_idxs, 3:7]

        # Set desired hand positions
        pos_des[:, 0] = 0 + math.cos(itr / 50) * 0.02
        pos_des[:, 1] = table_dims.y
        pos_des[:, 2] = init_pos[:, 2] + math.cos(itr / 50) * 0.2

        # Solve for control (Operational Space Control)
        m_inv = torch.inverse(mm)
        m_eef = torch.inverse(j_eef @ m_inv @ torch.transpose(j_eef, 1, 2))
        orn_cur /= torch.norm(orn_cur, dim=-1).unsqueeze(-1)
        
        orn_err = orientation_error(orn_des, orn_cur)
        
        pos_err = kp * (pos_des - pos_cur)

        dpose = torch.cat([pos_err, orn_err], -1)

        u = torch.transpose(j_eef, 1, 2) @ m_eef @ (kp * dpose).unsqueeze(-1) - kv * mm @ dof_vel

        # Set tensor action
        gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(u))

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
    