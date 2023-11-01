import abc

class BaseEnv(abc.ABC):
    '''
        An abstract base class for simulation environment.
    '''
    def __init__(self, mjmodel):
        self.mj_model = mjmodel

    @abc.abstractmethod
    def step(self, ctrl):
        '''
        take a step of simulation
        '''
        pass

    @abc.abstractmethod
    def reset(self):
        '''
        reset the simulator
        '''
        pass
    
    @abc.abstractmethod
    def close(self):
        """
        close the simulator
        """
        pass
