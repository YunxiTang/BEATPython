from dataclasses import dataclass


@dataclass
class Obstacle:
    obs_type: str = None
    obs_pos: list = []
    obs_orient: list = []
    size: list = []


class Env:
    """
    extract config for planner
    """

    def __init__(self, xml_path):
        self.xml_path = xml_path

    def get_obstacles(self):
        return []
