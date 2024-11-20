import sys

from ompl import base as ob
from ompl import geometric as og
from ompl import util as ou

import numpy as np
from st_dlo_planning.spatial_pathset_gen.world_map import WorldMap

import matplotlib.pyplot as plt


DEFAULT_PLANNING_TIME = 200.
INTERPOLATE_NUM = 40


class DloOmpl:
    def __init__(self, world_map: WorldMap, z: float, 
                 k_pathLen: float=1.0,
                 k_clearance: float=1.0,
                 k_passage: float=100.,
                 animation: bool=False):
        '''
        Args
            world_map: the map of the workspace with obstacles
        '''

        # world map will be used for stateValidator and so on
        self.world_map = world_map

        # ======== step 1: state space initialization (x \in R^{3}) =========
        self.state_space = ob.RealVectorStateSpace(3)

        bounds = ob.RealVectorBounds(3)
        bounds.setLow(0, world_map.map_cfg.map_xmin)
        bounds.setHigh(0, world_map.map_cfg.map_xmax)
        bounds.setLow(1, world_map.map_cfg.map_ymin)
        bounds.setHigh(1, world_map.map_cfg.map_ymax)
        bounds.setLow(2, z)
        bounds.setHigh(2, z)
        self.state_space.setBounds(bounds)

        # ======== step 2: create a simple setup ================
        self.simple_setup = og.SimpleSetup(self.state_space)
        
        # ======== step 3: get the space info ===================
        self.space_info = self.simple_setup.getSpaceInformation()
        
        # ======== step 4: set a state validality checker =======
        state_validality_checker = MyStateValidityChecker(self.space_info, self.world_map)
        self.space_info.setStateValidityChecker(state_validality_checker)
        
        # ======== step 5: set a validate state sampler =========
        # TODO: does not work as expected ?
        # return an instance of my sampler
        # def allocMyValidStateSampler(si):
        #     return MyValidStateSampler(si, self.world_map)
        
        # self.space_info.setValidStateSamplerAllocator(
        #     ob.ValidStateSamplerAllocator(allocMyValidStateSampler)
        #     )
        
        # ======== step 6: get the problem definition and callback_fn =======
        self.problem_def = self.simple_setup.getProblemDefinition()

        def call_back(planner, intermidiate_solution, cost):
            print(f'Cost: {cost.value()}')

            if animation:
                states = []
                for state in intermidiate_solution:
                    states.append([state[0], state[1], state[2]])
                    sol_np = np.array(states)

                ax = self.world_map.visualize_passage(full_passage=False)
                for i in range(len(states)-1):
                    ax.plot([sol_np[i, 0], sol_np[i+1, 0]], 
                            [sol_np[i, 1], sol_np[i+1, 1]], 'r-')
                plt.axis('equal')
                plt.show()
            
        self.problem_def.setIntermediateSolutionCallback(ob.ReportIntermediateSolutionFn(call_back))

        # ======== step 7: set the optimization objective
        pathLengthObjective = ob.PathLengthOptimizationObjective(self.space_info)
        passageWidthObjective = PassageOptimizationObjective(self.space_info, self.world_map)
        clearance_obj = ClearanceObjective(self.space_info)
        
        multiObjective = ob.MultiOptimizationObjective(self.space_info)
        multiObjective.addObjective(pathLengthObjective, k_pathLen)
        multiObjective.addObjective(passageWidthObjective, k_passage)
        multiObjective.addObjective(clearance_obj, k_clearance)
        
        self.simple_setup.setOptimizationObjective(multiObjective)

        # ======== step 7: set the planner ======================
        self.set_planner('RRTstar')

        
    def plan(self, 
             start: list, 
             goal: list, 
             allowed_time: float = DEFAULT_PLANNING_TIME,
             num_waypoints: int = INTERPOLATE_NUM):
        '''
            plan a path from start to goal
        '''
        print("start_planning")

        # set the start and goal states;
        s = ob.State(self.state_space)
        g = ob.State(self.state_space)
        for i in range(len(start)):
            s[i] = start[i]
            g[i] = goal[i]

        self.simple_setup.setStartAndGoalStates(s, g)

        # attempt to solve the problem within allowed planning time
        solved = self.simple_setup.solve(allowed_time)
        if solved:
            print("Found solution.")
            sol_path_geometric = self.simple_setup.getSolutionPath()
            sol_path_geometric.interpolate(num_waypoints)

            states = []
            for i in range(sol_path_geometric.getStateCount()):
                state = sol_path_geometric.getState(i)
                states.append([state[0], state[1], state[2]])
            sol_np = np.array(states)
            return sol_path_geometric, sol_np
        else:
            print('No solution can be found')
            return None, None
    
    
    def set_planner(self, planner_name: str = 'RRTstar'):
        '''
            args:
                planner_name (str): a planner name
            Set the planner to be used.
            Note: you can add any customized planner here.
        '''
        if planner_name == "PRM":
            self.planner = og.PRM(self.simple_setup.getSpaceInformation())
        elif planner_name == "RRT":
            self.planner = og.RRT(self.simple_setup.getSpaceInformation())
        elif planner_name == "RRTConnect":
            self.planner = og.RRTConnect(self.simple_setup.getSpaceInformation())
        elif planner_name == "RRTstar":
            self.planner = og.RRTstar(self.simple_setup.getSpaceInformation())
        elif planner_name == "EST":
            self.planner = og.EST(self.simple_setup.getSpaceInformation())
        elif planner_name == "FMT":
            self.planner = og.FMT(self.simple_setup.getSpaceInformation())
        elif planner_name == "BITstar":
            self.planner = og.BITstar(self.simple_setup.getSpaceInformation())
        else:
            print("{} not recognized, please add it first".format(planner_name))
            return

        self.simple_setup.setPlanner(self.planner)
        
    def _state_to_list(self, state):
        return [state[i] for i in range(self.state_space.getDimension())]
    
    
# ======================== customized moddules ===============
# ValidateStateSampler
class MyValidStateSampler(ob.ValidStateSampler):
    def __init__(self, space_info, world_map: WorldMap):
        super(MyValidStateSampler, self).__init__(space_info)
        self.name_ = "validate_state_sampler"
        self.rng_ = ou.RNG()
        self.world_map = world_map

    # Generate a sample in the valid part of a state space.
    def sample(self, state):
        z = self.world_map.sample_validate_position()
        state[0] = z[0]
        state[1] = z[1]
        state[2] = z[2]
        return True
    

# state validality checker
class MyStateValidityChecker(ob.StateValidityChecker):
    def __init__(self, space_info, world_map: WorldMap):
        super(MyStateValidityChecker, self).__init__(space_info)
        self.world_map = world_map

    def isValid(self, state):
        pos = np.array([state[0], state[1], state[2]])
        validate = self.world_map.check_pos_collision(pos)
        return validate
    
    def clearance(self, state):
        # compute the clearance
        clearance = self.world_map.compute_clearance(state)
        return clearance


class ClearanceObjective(ob.StateCostIntegralObjective):
    def __init__(self, si):
        super(ClearanceObjective, self).__init__(si, True)
        self.si_ = si

    def stateCost(self, state):
        clearance = self.si_.getStateValidityChecker().clearance(state)
        cost_val = 1. / (  clearance ** 2 )
        return ob.Cost( cost_val )
    

class PassageOptimizationObjective(ob.OptimizationObjective):
    '''
        Passage-aware objective (to get wider passage width).
        This optimization objective is implemented based on the implementation of {maximize minimum clearance}
    '''
    def __init__(self, space_info, world_map: WorldMap):
        super(PassageOptimizationObjective, self).__init__(space_info)
        self.space_info = space_info
        self.world_map = world_map

    def _get_array_from_state(self, state):
        return np.array([state[0], state[1], state[2]])

    def stateCost(self, state):
        '''
            query the passage width from the world map
        '''
        return ob.Cost(1.0)
    
    def motionCost(self, state1, state2):
        '''
            motion->parent = nmotion;
            motion->incCost = opt_->motionCost(nmotion->state, motion->state);
            motion->cost = opt_->combineCosts(nmotion->cost, motion->incCost);
        '''
        state1_np = self._get_array_from_state(state1)
        state2_np = self._get_array_from_state(state2)
        passage = self.world_map.check_passage_intersection(state1_np, state2_np)
        
        if passage is not None:
            passage_width = passage.min_dist
            incremental_cost = 1. / (passage_width**2)
        else:
            incremental_cost = 1e-5
        return ob.Cost(incremental_cost)
    
    def combineCosts(self, cost1, cost2):
        '''
            cost1: parent node cost.
            cost2: the incremental cost of this motion's parent to this motion
        '''
        if (cost1 > cost2.value()):
            return cost1
        else:
            return cost2
        # return cost1 + cost2
    
    # def isCostBetterThan(self, cost1, cost2):
    #     return cost1 < cost2 + 1e-3

    