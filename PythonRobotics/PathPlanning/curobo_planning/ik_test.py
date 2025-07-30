# Third Party
import torch

# cuRobo
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_path, join_path, load_yaml

# convenience function to store tensor type and device
tensor_args = TensorDeviceType()

# this example loads urdf from a configuration file, you can also load from path directly
# load a urdf, the base frame and the end-effector frame:
config_file = load_yaml(join_path(get_robot_path(), "franka.yml"))

urdf_file = config_file["robot_cfg"]["kinematics"][
    "urdf_path"
]  # Send global path starting with "/"
base_link = config_file["robot_cfg"]["kinematics"]["base_link"]
ee_link = config_file["robot_cfg"]["kinematics"]["ee_link"]

# Generate robot configuration from  urdf path, base frame, end effector frame

robot_cfg = RobotConfig.from_basic(urdf_file, base_link, ee_link, tensor_args)

kin_model = CudaRobotModel(robot_cfg.kinematics)

# compute forward kinematics:
# torch random sampling might give values out of joint limits
q = torch.rand((10, kin_model.get_dof()), **(tensor_args.as_torch_dict()))
out = kin_model.get_state(q)

print(out)