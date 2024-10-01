if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    print(ROOT_DIR)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

from components import Node, State, Modality, Terrian, WorldMap
from modality_rrt_star import ModalRRTStar

if __name__ == '__main__':
    print('modality planning')
    
    terrian_map = WorldMap()
    
    start_cfg = Node(state=State(0.0, Modality.Rolling))
    goal_cfg = Node(state=State(30., Modality.Rolling))
    planner = ModalRRTStar(
        connect_range=1.0,
        start_config=start_cfg,
        goal_config=goal_cfg,
        map=terrian_map,
        step_size = 0.2
    )
    planner.plan()