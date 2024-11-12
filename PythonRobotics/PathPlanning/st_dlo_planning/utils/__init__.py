from dataclasses import dataclass
from path_smooth import smooth_trajectory
from compute_energy import compute_enery

@dataclass
class ENV_CFG:
    range: None
    obstacle: None
    goal: None
    start: None
    
    
class PathSet:
    def __init__(self, paths, T:int):
        self.paths = paths
        self.num_path = len(paths)
        self.T = T
        
    @property
    def T(self):
        return self.T
    
    @property
    def paths(self):
        return self.paths