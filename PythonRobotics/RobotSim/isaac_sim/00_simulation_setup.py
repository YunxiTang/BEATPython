from isaacgym import gymapi
import numpy as np
import os


if __name__ == '__main__':
    gym = gymapi.acquire_gym()

    # create a simulation with sim_params
    sim_params = gymapi.SimParams()
    sim_params.dt = 1 / 60.
    sim_params.substeps = 2
    sim_params.use_gpu_pipeline = True
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

    # set PhysX-specific parameters
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

    # load an asset
    asset_root = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),  'assets')
    asset_file = 'urdf/ycb/025_mug/025_mug.urdf'
    
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = False
    asset_options.disable_gravity = False

    mug_asset = gym.load_asset(
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
    env_spacing = 1

    lower = gymapi.Vec3(-env_spacing, 0.0,   -env_spacing)
    upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

    env_handles = []
    mug_handles = []

    for i in range(env_num):
        # create env
        env = gym.create_env(sim, lower, upper, env_per_row)
        env_handles.append(env)

        # create actor
        mug_pose = gymapi.Transform()
        mug_pose.p = gymapi.Vec3(0.0, 0.0, 1.0)
        zyx = 2 * np.random.random(3)
        mug_pose.r = gymapi.Quat.from_euler_zyx(zyx[0], zyx[1], zyx[2]) 

        mug_handle = gym.create_actor(env, mug_asset, mug_pose, "mug", i, 0)

        color = np.random.random(3)
        gym.set_rigid_body_color(env, mug_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(color[0], color[1], color[2]))
        mug_handles.append(mug_handle)

        origin_pos = gym.get_env_origin(env)
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




