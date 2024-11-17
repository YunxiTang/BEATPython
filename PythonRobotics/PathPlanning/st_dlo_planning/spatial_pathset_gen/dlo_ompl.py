from ompl import base as ob
from ompl import geometric as og
from ompl import util as ou

import numpy as np
from st_dlo_planning.spatial_pathset_gen.world_map import WorldMap


DEFAULT_PLANNING_TIME = 10.
INTERPOLATE_NUM = 20


 # return an instance of my sampler
def allocMyValidStateSampler(si):
    return MyValidStateSampler(si)


class DloOmpl:
    def __init__(self, world_map: WorldMap):
        '''
        Args
            world_map: the map of the workspace with obstacles
        '''
        # world map will be used for stateValidator and so on
        self.world_map = world_map

        # ======== step 1: state space initialization (x \in R^{3}) =========
        self.state_space = ob.RealVectorStateSpace(3)

        bounds = ob.RealVectorBounds(3)
        bounds.setLow(-1)
        bounds.setHigh(1)

        self.state_space.setBounds(bounds)

        # ======== step 2: create a simple setup ================
        self.simple_setup = og.SimpleSetup(self.state_space)
        
        # ======== step 3: get the space info ===================
        self.space_info = self.simple_setup.getSpaceInformation()
        
        # ======== step 4: set a state validality checker =======
        state_validality_checker = MyStateValidityChecker()
        self.space_info.setStateValidityChecker(state_validality_checker)
        
        # ======== step 5: set a validate state sampler =========
        self.space_info.setValidStateSamplerAllocator(
            ob.ValidStateSamplerAllocator(allocMyValidStateSampler)
            )
        
        # ======== step 6: set the optimization objective =======

        # ======== step 7: set the planner ======================
        self.set_planner('RRTstar')
        
    def is_state_valid(self, state):
        # check collision against environment based on the world map
        pos = np.array([state[0], state[1]])
        validtate = self.world_map.check_pos_collision(pos)
        return validtate
        
    def plan(self, 
             start: list, 
             goal: list, 
             allowed_time: float = DEFAULT_PLANNING_TIME):
        '''
            plan a path from start to goal
        '''
        print("start_planning")

        # set the start and goal states;
        s = ob.State(self.space)
        g = ob.State(self.space)
        for i in range(len(start)):
            s[i] = start[i]
            g[i] = goal[i]

        self.simple_setup.setStartAndGoalStates(s, g)

        # attempt to solve the problem within allowed planning time
        solved = self.simple_setup.solve(allowed_time)
        if solved:
            print("Found solution: interpolating into {} segments".format(INTERPOLATE_NUM))
            # print the path to screen
            sol_path_geometric = self.simple_setup.getSolutionPath()
            return sol_path_geometric
        else:
            print('No solution can be found')
            None

    def stateValidator(self, state):
        pass


    def objectiveFn(self, state):
        pass
    
    
    def set_planner(self, planner_name: str = 'RRTstar'):
        '''
            args:
                planner_name (str): a planner name
            Set the planner to be used.
            Note: you can add any customized planner here.
        '''
        if planner_name == "PRM":
            self.planner = og.PRM(self.ss.getSpaceInformation())
        elif planner_name == "RRT":
            self.planner = og.RRT(self.ss.getSpaceInformation())
        elif planner_name == "RRTConnect":
            self.planner = og.RRTConnect(self.ss.getSpaceInformation())
        elif planner_name == "RRTstar":
            self.planner = og.RRTstar(self.ss.getSpaceInformation())
        elif planner_name == "EST":
            self.planner = og.EST(self.ss.getSpaceInformation())
        elif planner_name == "FMT":
            self.planner = og.FMT(self.ss.getSpaceInformation())
        elif planner_name == "BITstar":
            self.planner = og.BITstar(self.ss.getSpaceInformation())
        else:
            print("{} not recognized, please add it first".format(planner_name))
            return

        self.simple_setup.setPlanner(self.planner)
        
    def _state_to_list(self, state):
        return [state[i] for i in range(self.state_space.getDimension())]
    
    
# ======================== customized moddules ===============
# validate state sampler
class MyValidStateSampler(ob.ValidStateSampler):
    def __init__(self, space_info, world_map: WorldMap):
        super(MyValidStateSampler, self).__init__(space_info)
        self.name_ = "validate state sampler"
        self.rng_ = ou.RNG()
        self.world_map = world_map

    # Generate a sample in the valid part of a state space.
    def sample(self, state):
        pos = self._get_pos_from_state(state)
        validtate = self.world_map.check_pos_collision(pos)
        return validtate
    
    def _get_pos_from_state(self, state):
        return np.array([state[0], state[1]])
    
class MyStateValidityChecker:
    def __init__(self):
        pass
    
    
class MyOptimObjective:
    def __init__(self):
        pass