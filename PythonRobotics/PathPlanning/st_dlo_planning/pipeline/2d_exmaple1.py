from ..utils import ENV_CFG

# step 1: obtain env_cfg for path set generation
env_cfg = ENV_CFG()

# step 2: obtain the start and goal DLO configs
start_dlo_cfg = None
goal_dlo_cfg = None

# step 3: choose the pivolt point for path planning
pivolt_start = None
pivolt_goal = None

pivolt_optimal_path = None

# step 4: transfer the pivolt path and get the pathset
pathset = None

# step 5: optimize the intermediate DLO confiurations
dlo_config_seq = []


# step 6: track the intermediate DLO confiurations in hardware or simulator
# step 6-1: load GDM for MPC
# step 6-2: setup MPC
# step 6-3: run

