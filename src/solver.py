from enum import Enum

import numpy as np

from abc import abstractmethod, ABCMeta
from src.ivp import IVP
from src.solution import Solution


class MethodType(Enum):
    explicit = 1
    implicit = 2
    predictor_corrector = 3
    unspecified = 4



class Solver(object):
    """ Class representing an abstract numerical solver (method or group of
        methods) for a given initial value problem
    """

    # abstract class definition
    __metaclass__ = ABCMeta


    # type hints for instance variables
    ivp: IVP
    current_time: float
    end_time: float
    method_type: MethodType


    def __init__(self, ivp, end_time):
        """ Initialising instance variables """
        # simple setting of input values
        self.ivp = ivp
        self.current_time = self.ivp.initial_time
        self.end_time = end_time

        # produces empty time and value meshes
        self.time_mesh = self.build_time_mesh()
        self.value_mesh = self.build_value_mesh(self.ivp.get_dimension())

        # creates empty solution object for print method
        self.solution = Solution(self.time_mesh, self.value_mesh)


    def build_time_mesh(self):
        return np.zeros(self.max_mesh_size())


    def build_value_mesh(self, dimension):
        return np.zeros((self.max_mesh_size(), dimension))


    def print_solution(self):
        print(self.solution)


    @abstractmethod
    def solve(self):
        """ Running approximation up until end_time """
        pass


    @abstractmethod
    def forward_step(self, step_counter, call_from=MethodType.unspecified, u_prediction=None):
        """ Adjusting instance variables and calculating the next step size
            before performing next iteration.
        """
        pass


    @abstractmethod
    def max_mesh_size(self):
        """ Calculates the biggest dimension of mesh for this Solver, given the
            maximum step size and final time value
        """
        pass

    @abstractmethod
    def next_step_size(self, this_step):
        """ Return the step size for the next iteration of the method. """
        pass
