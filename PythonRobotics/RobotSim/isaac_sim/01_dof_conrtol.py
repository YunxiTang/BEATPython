import isaacgym
from isaacgym import gymapi, gymutil, gymtorch
import numpy as np
import torch
import os


if __name__ == '__main__':
    # create a gym space
    gym = gymapi.acquire_gym()

    # create sim_params
    sim_params = gymapi.SimParams()
    sim_params.dt = 1 / 60.
    sim_params.substeps = 2
    sim_params.use_gpu_pipeline = True
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

    # set PhysX engine parameters
    sim_params.physx.use_gpu = True
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 6
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.contact_offset = 0.01
    sim_params.physx.rest_offset = 0.0

    compute_device = 0
    graphics_device = 0
    sim = gym.create_sim(compute_device, 
                         graphics_device, 
                         gymapi.SIM_PHYSX, 
                         sim_params)
    
    # add a ground plane
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)

    gym.add_ground(sim, plane_params)

    # load ur5e asset
    asset_root = '/home/yxtang/CodeBase/PythonCourse/PythonRobotics/RobotSim/isaac_sim/assets/urdf/ur_e_description'
    asset_file = 'universalUR5e.urdf'
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.disable_gravity = False
    asset_options.use_mesh_materials = True

    ur5_asset = gym.load_asset(
        sim,
        asset_root,
        asset_file,
        asset_options
    )

    # configure franka dofs
    ur5_dof_props = gym.get_asset_dof_properties(ur5_asset)

    print(ur5_dof_props)
    
    ur5_dof_props["driveMode"][:].fill(gymapi.DOF_MODE_POS)
    ur5_dof_props['stiffness'][:].fill(400.0)
    ur5_dof_props['damping'][:].fill(40.0)
        

    print("DOF Properties:")
    for i, props in enumerate(ur5_dof_props):
        print(f"DOF {i}:")
        print(f"  Drive Mode: {props['driveMode']}")
        print(f"  Stiffness: {props['stiffness']}")
        print(f"  Damping: {props['damping']}")
        print(f"  Effort: {props['effort']}")
        print(f"  Velocity: {props['velocity']}")
        print(f"  Friction: {props['friction']}")
        print(f"  Lower Limit: {props['lower']}")
        print(f"  Upper Limit: {props['upper']}")
        print(f"  Has Limits: {props['hasLimits']}")

    
    # get link index of panda hand, which we will use as end effector
    ur5e_link_dict = gym.get_asset_rigid_body_dict(ur5_asset)
    
    eef_link_index = ur5e_link_dict["tool0"] # 8
    
    # load table asset
    asset_root = '/home/yxtang/CodeBase/PythonCourse/PythonRobotics/RobotSim/isaac_sim/assets/urdf'
    asset_file = 'square_table.urdf'
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.disable_gravity = False
    asset_options.use_mesh_materials = True

    table_asset = gym.load_asset(
        sim,
        asset_root,
        asset_file,
        asset_options
    )

    # create viewer
    camera_properties = gymapi.CameraProperties()
    viewer = gym.create_viewer(sim, camera_properties)

    env_num = 16
    env_per_row = 4
    env_spacing = 2

    lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
    upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

    env_handles = []
    ur5_handles = []

    for i in range(env_num):
        # create env
        env_handle = gym.create_env(sim, lower, upper, env_per_row)
        env_handles.append(env_handle)

        # ============== create actor for env ======================================
        # one table
        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0, 1.5, 0.5)
        zyx = 2 * np.random.random(3) * 0
        table_pose.r = gymapi.Quat.from_euler_zyx(zyx[0], zyx[1], zyx[2]) 
        table_handle = gym.create_actor(env_handle, table_asset, table_pose, "table", i, 0)

        # two ur5e robots
        ur5_pose_1 = gymapi.Transform()
        ur5_pose_1.p = gymapi.Vec3(-0.75, 1.5, 1.56)
        zyx = 2 * np.random.random(3) * 0
        ur5_pose_1.r = gymapi.Quat.from_euler_zyx(zyx[0], zyx[1], zyx[2]) 
        ur5_handle_1 = gym.create_actor(env_handle, ur5_asset, ur5_pose_1, "ur5e1", i, 0)
        gym.set_actor_dof_properties(env_handle, ur5_handle_1, ur5_dof_props)

        ur5_pose_2 = gymapi.Transform()
        ur5_pose_2.p = gymapi.Vec3(0.75, 1.5, 1.56)
        zyx = 2 * np.random.random(3) * 0
        ur5_pose_2.r = gymapi.Quat.from_euler_zyx(zyx[0], zyx[1], np.pi*0) 
        ur5_handle_2 = gym.create_actor(env_handle, ur5_asset, ur5_pose_2, "ur5e2", i, 0)
        gym.set_actor_dof_properties(env_handle, ur5_handle_2, ur5_dof_props)

        # color = [0.2, 0.2, 0.8] #np.random.random(3)
        # gym.set_rigid_body_color(env_handle, ur5_handle, 3, gymapi.MESH_VISUAL, gymapi.Vec3(color[0], color[1], color[2]))
        # ur5_handles.append(ur5_handle_1)

        origin_pos = gym.get_env_origin(env_handle)
        print(origin_pos)
    
    gym.prepare_sim(sim)

    body_num = gym.get_asset_rigid_body_count(ur5_asset)

    rigid_body_state_desc = gym.acquire_rigid_body_state_tensor(sim)
    rigid_body_state_tensor = gymtorch.wrap_tensor(rigid_body_state_desc)
    rigid_body_positions = rigid_body_state_tensor[:, 0:3].view(env_num, body_num * 2 + 1, 3)
    rigid_body_velocities = rigid_body_state_tensor[:, 7:13].view(env_num, body_num * 2 + 1, 6)

    dof_state_desc = gym.acquire_dof_state_tensor(sim)
    dof_state_tensor = gymtorch.wrap_tensor(dof_state_desc)
    dof_state_tensor = dof_state_tensor.view(env_num, -1, 2)

    ur5_1_jacobians_desc = gym.acquire_jacobian_tensor(sim, 'ur5e1')
    ur5_1_jacobians_tensor = gymtorch.wrap_tensor(ur5_1_jacobians_desc) # in shape of: (num_envs, 10, 6, num_dofs)
    ur5e_1_eef_jacobian_tensor = ur5_1_jacobians_tensor[:, eef_link_index-1, :, :]
   
    interval = 300
    steps = 0
    while not gym.query_viewer_has_closed(viewer):
        # apply action
        if steps % interval == 0:
            offsets = torch.randn([env_num, 12], device='cuda:0') * 1.0 
        target_jnt_pos = offsets + 1 * torch.sin(torch.tensor(steps * 1/ 60, device='cuda:0'))
        gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(target_jnt_pos))

        # step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.refresh_actor_root_state_tensor(sim)
        gym.refresh_rigid_body_state_tensor(sim)
        gym.refresh_dof_state_tensor(sim)
        gym.refresh_jacobian_tensors(sim)
        
        # update the viewer
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

        # check Jacobian's correctness
        dof_velocity = dof_state_tensor[:, 0:6, 1:]
        eef_velocity = torch.bmm(ur5e_1_eef_jacobian_tensor, dof_velocity)
        eef_velocity_gt = rigid_body_velocities[0, 10].clone()
        print(torch.linalg.norm(eef_velocity[0].flatten() - eef_velocity_gt))
        
        print(f'Sim. Step: {steps}')

        steps += 1
    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)




