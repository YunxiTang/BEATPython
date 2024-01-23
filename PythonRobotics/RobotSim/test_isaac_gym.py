from isaacgym import gymapi
import numpy as np


if __name__ == '__main__':
    gym = gymapi.acquire_gym()
    # set device
    compute_device_id = 'cuda:0'
    graphics_device_id = 'cuda:0'

    # ============= create a simulation =======================
    # get default set of parameters
    sim_params = gymapi.SimParams()
    # set common parameters
    sim_params.dt = 1 / 60
    sim_params.substeps = 2
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

    # set PhysX-specific parameters
    sim_params.physx.use_gpu = True
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 6
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.contact_offset = 0.01
    sim_params.physx.rest_offset = 0.0
    
    sim = gym.create_sim(compute_device_id, graphics_device_id, gymapi.SIM_PHYSX, sim_params)
