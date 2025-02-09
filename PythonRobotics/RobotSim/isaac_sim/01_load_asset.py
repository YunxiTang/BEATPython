import isaacgym
from isaacgym import gymapi, gymutil, gymtorch
import torch
import os


class Env:
    def __init__(self, env_dist, dt):
        # get gym handle
        self.gym = gymapi.acquire_gym()

        self.sim_params = gymapi.SimParams()
        self.sim_params.dt = dt
        self.sim_params.substeps = 2
        self.sim_params.use_gpu_pipeline = True

        # set PhysX engine parameters
        self.sim_params.physx.use_gpu = True

        compute_device = 0
        graphics_device = 0

        # get sim handle
        self.sim = self.gym.create_sim(compute_device, graphics_device, gymapi.SIM_PHYSX, self.sim_params)

        # add a ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)

        # load an asset
        asset_root = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),  'assets')
        asset_file = 'urdf/ycb/025_mug/025_mug.urdf'# 'urdf/ur5/ur5.urdf'
        
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.disable_gravity = False

        mug_asset = self.gym.load_asset(
            self.sim,
            asset_root,
            asset_file,
            asset_options
        )

        # create viewer
        camera_properties = gymapi.CameraProperties()
        self.viewer = self.gym.create_viewer(self.sim, camera_properties)

    def step():
        

        

if __name__ == '__main__':
    env = Env(dt=1/60)
