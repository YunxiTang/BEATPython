import isaacgym
from isaacgym import gymapi, gymutil, gymtorch, torch_utils
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt


def quat_error(desired_quat, current_quat):
    current_conjugate = torch_utils.quat_conjugate(current_quat)
    relative_quat = torch_utils.quat_mul(desired_quat, current_conjugate)
    return 2*relative_quat[:, 0:3] * torch.sign(relative_quat[:, 3]).unsqueeze(-1)

def get_ik_control(dpose, eef_jacobian, damping):
    # solve damped least squares
    j_eef_T = torch.transpose(eef_jacobian, 1, 2)
    lmbda = torch.eye(6, device=eef_jacobian.device) * (damping ** 2)
    u = (j_eef_T @ torch.inverse(eef_jacobian @ j_eef_T + lmbda) @ dpose).view(-1, 6)
    return u


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
    ur5e_ndof = len(ur5_dof_props)
    
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
    asset_options.disable_gravity = True
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
    rigid_body_positions = rigid_body_state_tensor[:, 0:7].view(env_num, body_num * 2 + 1, 7)
    rigid_body_velocities = rigid_body_state_tensor[:, 7:13].view(env_num, body_num * 2 + 1, 6)

    dof_state_desc = gym.acquire_dof_state_tensor(sim)
    dof_state_tensor = gymtorch.wrap_tensor(dof_state_desc)
    dof_state_tensor = dof_state_tensor.view(env_num, -1, 2)

    ur5_1_jacobians_desc = gym.acquire_jacobian_tensor(sim, 'ur5e1')
    ur5_1_jacobians_tensor = gymtorch.wrap_tensor(ur5_1_jacobians_desc) # in shape of: (num_envs, 10, 6, num_dofs)
    ur5e_1_eef_jacobian_tensor = ur5_1_jacobians_tensor[:, eef_link_index-1, :, :]
   
    steps = 0

    target_pos_0 = torch.stack(env_num * [torch.tensor([-0.75, 1.5, 2.2])]).to('cuda:0').view((env_num, 3))
    target_quat_s = gymapi.Quat.from_euler_zyx(0, 0, np.pi/4) 
    target_quat = torch.stack(env_num * [torch.tensor([target_quat_s.x, target_quat_s.y, target_quat_s.z, target_quat_s.w])]).to('cuda:0').view((env_num, 4)) # x y z w
    # target_pose = torch.concatenate((target_pos, target_quat), dim=1)
    
    target_jnt_pos = torch.randn([env_num, 12], device='cuda:0') * 0.0 

    # global rigid body index for ur5e_1 tool
    ur5e_1_eef_body_index = []
    env = env_handles[0]
    actor_handle = gym.get_actor_handle(env, 1)
    # 获取该 Actor 内的刚体数量
    num_rigid_bodies = gym.get_actor_rigid_body_count(env, actor_handle)
    # 获取eef的local index
    rigid_body_index = gym.get_actor_rigid_body_index(env, actor_handle, num_rigid_bodies-1, gymapi.DOMAIN_ENV)
    ur5e_1_eef_body_index = rigid_body_index

    reference_traj = []
    tracking_res = []

    max_step = 2000
    while not gym.query_viewer_has_closed(viewer):
        offset = torch.tensor([0.01 * np.cos(steps*0.02), 
                               0.01 * np.sin(steps*0.02), 
                               0.0], device='cuda:0', dtype=torch.float32)[None]
        target_pos = target_pos_0 + offset

        # step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        gym.refresh_actor_root_state_tensor(sim)
        gym.refresh_rigid_body_state_tensor(sim)
        gym.refresh_dof_state_tensor(sim)
        gym.refresh_jacobian_tensors(sim)

        # apply action
        eef_pose = rigid_body_positions[:, ur5e_1_eef_body_index, :].clone()
        delta_pos = target_pos - eef_pose[:, 0:3]
        delta_quat = quat_error(target_quat, eef_pose[:, 3:])
        dpose = torch.concatenate((delta_pos, delta_quat), dim=1).unsqueeze(-1)

        reference_traj.append(target_pos[0].cpu().numpy())
        tracking_res.append(eef_pose[0].cpu().numpy())
        print(dpose[0])

        # delta_jnt_pos = torch.clamp(get_ik_control(dpose, ur5e_1_eef_jacobian_tensor, 0.05),
        #                             min=torch.tensor([[-0.2]*6], device='cuda:0'),
        #                             max=torch.tensor([[ 0.2]*6], device='cuda:0'))
        delta_jnt_pos = get_ik_control(dpose, ur5e_1_eef_jacobian_tensor, 1.1)
        target_jnt_pos[:, 0:6] = target_jnt_pos[:, 0:6] + 0.1 * delta_jnt_pos.squeeze()

        gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(target_jnt_pos))
        
        # update the viewer
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

        # check Jacobian's correctness
        dof_velocity = dof_state_tensor[:, 0:6, 1:]
        eef_velocity = torch.bmm(ur5e_1_eef_jacobian_tensor, dof_velocity)
        eef_velocity_gt = rigid_body_velocities[0, ur5e_1_eef_body_index].clone()
        print(torch.linalg.norm(eef_velocity[0].flatten() - eef_velocity_gt))
        
        print(f'Sim. Step: {steps}')

        steps += 1

        if steps > max_step:
            break

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

    sns.set_theme('notebook')
    xs = [res[0] for res in tracking_res]
    ys = [res[1] for res in tracking_res]
    plt.plot(xs, ys)
    xs = [res[0] for res in reference_traj]
    ys = [res[1] for res in reference_traj]
    plt.plot(xs, ys)
    plt.axis('equal')
    plt.show()




