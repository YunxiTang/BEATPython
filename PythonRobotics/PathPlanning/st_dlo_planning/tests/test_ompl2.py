from ompl import geometric as og
from ompl import base as ob
from ompl import util as ou
import math


def isStateValid(state):
    # Some arbitrary condition on the state (note that thanks to
    # dynamic type checking we can just call getX() and do not need
    # to convert state to an SE2State.)
    return state.getX() < 0.6


def planWithSimpleSetup():
    # create an SE2 state space
    space = ob.SE2StateSpace()

    # set lower and upper bounds
    bounds = ob.RealVectorBounds(2)
    bounds.setLow(-1)
    bounds.setHigh(1)
    space.setBounds(bounds)

    # create a simple setup object
    ss = og.SimpleSetup(space)
    ss.setStateValidityChecker(ob.StateValidityCheckerFn(isStateValid))

    start = ob.State(space)
    # we can pick a random start state...
    start.random()
    # ... or set specific values
    start().setX(0.5)

    goal = ob.State(space)
    # we can pick a random goal state...
    goal.random()
    # ... or set specific values
    goal().setX(-0.5)

    ss.setStartAndGoalStates(start, goal)

    print(start, goal)

    # this will automatically choose a default planner with
    # default parameters
    solved = ss.solve(1.0)

    if solved:
        # try to shorten the path
        ss.simplifySolution()
        # print the simplified path
        print(ss.getSolutionPath())


def planWithR2():
    # create a state space
    state_space = ob.RealVectorStateSpace(2)

    # set lower and upper bounds
    bounds = ob.RealVectorBounds(2)
    bounds.setLow(-1)
    bounds.setHigh(1)

    state_space.setBounds(bounds)

    # create a simple setup object
    simple_setup = og.SimpleSetup(state_space)

    def isStateValid(state):
        return state[1] < 0.7

    simple_setup.setStateValidityChecker(ob.StateValidityCheckerFn(isStateValid))

    start = ob.State(state_space)
    # we can pick a random start state...
    start.random()

    print(start)
    # ... or set specific values
    start[1] = 0.6
    print(start)

    goal = ob.State(state_space)
    # we can pick a random goal state...
    goal.random()
    # ... or set specific values
    goal[1] = -0.5

    simple_setup.setStartAndGoalStates(start, goal)

    print(start, goal)

    # this will automatically choose a default planner with
    # default parameters
    solved = simple_setup.solve(1.0)

    if solved:
        # try to shorten the path
        simple_setup.simplifySolution()
        # print the simplified path
        print(simple_setup.getSolutionPath())


if __name__ == "__main__":
    planWithR2()
    # planWithSimpleSetup()
