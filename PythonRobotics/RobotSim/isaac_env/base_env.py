from isaacgym import gymapi
from isaacgym import gymutil
import numpy as np
import random


class BaseEnv:
    def __init__(self, config) -> None:
        self._gym = gymapi.acquire_gym()
        self._args = gymutil.parse_arguments(description="Isaac Gym Env")

        # ============= create a simulation =======================
        # get default set of parameters
        self._sim_params = gymapi.SimParams()

        # set common parameters
        self._sim_params.dt = config.dt
        self._sim_params.substeps = config.substeps
        self._sim_params.up_axis = gymapi.UP_AXIS_Z
        self._sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

        # set PhysX-specific parameters
        self._sim_params.physx.use_gpu = config.use_gpu
        self._sim_params.physx.solver_type = config.solver_type
        self._sim_params.physx.num_position_iterations = 6
        self._sim_params.physx.num_velocity_iterations = 1
        self._sim_params.physx.contact_offset = 0.01
        self._sim_params.physx.rest_offset = 0.0