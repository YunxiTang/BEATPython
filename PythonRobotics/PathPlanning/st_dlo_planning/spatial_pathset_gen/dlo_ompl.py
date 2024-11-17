from ompl import base as ob
from ompl import geometric as og
from ompl import util as ou

import numpy as np


class DloOmpl:
    def __init__(self, world_map):
        '''
        Args
            world_map: the map of the workspace with obstacles
        '''
        self.world_map = world_map

        self.state_space = ob.RealVectorStateSpace(2)

        # set lower and upper bounds
        bounds = ob.RealVectorBounds(2)
        bounds.setLow(-1)
        bounds.setHigh(1)

        self.state_space.setBounds(bounds)

        self.simple_setup = og.SimpleSetup(self.space)
        self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(self.is_state_valid))
        self.si = self.ss.getSpaceInformation()

    def stateValidator(self, state):
        pass


    def objectiveFn(self, state):
        pass