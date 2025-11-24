import isaacgym
from isaacgym import gymapi, gymutil, gymtorch
import numpy as np
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
    print("DOF Properties:")
    for i, props in enumerate(ur5_dof_props):
        print(f"DOF {i}:")
        print(f"  Stiffness: {props['stiffness']}")
        print(f"  Damping: {props['damping']}")
        print(f"  Effort: {props['effort']}")
        print(f"  Velocity: {props['velocity']}")
        print(f"  Friction: {props['friction']}")
        print(f"  Lower Limit: {props['lower']}")
        print(f"  Upper Limit: {props['upper']}")
        print(f"  Has Limits: {props['hasLimits']}")

    for props in ur5_dof_props:
        props['stiffness'] = 1000.0
        props['damping'] = 100.0
        
        # ur5_lower_limits = ur5_dof_props["lower"]
        # ur5_upper_limits = ur5_dof_props["upper"]
        # ur5_ranges = ur5_upper_limits - ur5_lower_limits
        # ur5_mids = 0.3 * (ur5_upper_limits + ur5_lower_limits)

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
        ur5_pose_1.p = gymapi.Vec3(-0.75, 1.5, 0.56)
        zyx = 2 * np.random.random(3) * 0
        ur5_pose_1.r = gymapi.Quat.from_euler_zyx(zyx[0], zyx[1], zyx[2]) 
        ur5_handle_1 = gym.create_actor(env_handle, ur5_asset, ur5_pose_1, "ur5e1", i, 0)

        ur5_pose_2 = gymapi.Transform()
        ur5_pose_2.p = gymapi.Vec3(0.75, 1.5, 0.56)
        zyx = 2 * np.random.random(3) * 0
        ur5_pose_2.r = gymapi.Quat.from_euler_zyx(zyx[0], zyx[1], np.pi*0) 
        ur5_handle_2 = gym.create_actor(env_handle, ur5_asset, ur5_pose_2, "ur5e2", i, 0)

        # color = [0.2, 0.2, 0.8] #np.random.random(3)
        # gym.set_rigid_body_color(env_handle, ur5_handle, 3, gymapi.MESH_VISUAL, gymapi.Vec3(color[0], color[1], color[2]))
        # ur5_handles.append(ur5_handle_1)

        origin_pos = gym.get_env_origin(env_handle)
        print(origin_pos)
    
    gym.prepare_sim(sim)

    while not gym.query_viewer_has_closed(viewer):
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




