from dataclasses import dataclass


@dataclass
class MapCfg:
    resolution: float = 0.1
    map_xmin: float = 0.
    map_xmax: float = 200.
    map_ymin: float = 0.
    map_ymax: float = 200.
    
    map_zmin: float = 99.9
    map_zmax: float = 100.1
    
    robot_size: float = 0.1
    
    dim: int = 3
    
    
    
if __name__ == '__main__':
    from pprint import pprint
    cfg = MapCfg()
    print(cfg.resolution)