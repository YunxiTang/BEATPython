if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    print(ROOT_DIR)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

from components import Node, State, Modality, Terrian
from modality_rrt_star import ModalRRTStar

if __name__ == '__main__':
    print('modality planning')
    print(Modality.Rolling)