from enum import Enum

import numpy as np

from abc import abstractmethod, ABCMeta
from src.ivp import IVP
from src.solution import Solution
from util.timelogging import TimedSolver


class MethodType(Enum):
    explicit = 1
    implicit = 2
    predictor_corrector = 3
    unspecified = 4



class Solver(TimedSolver):
    """ Class representing an abstract numerical solver (method or group of
        methods) for a given initial value problem
    """

    # abstract class definition
    __metaclass__ = ABCMeta


    # type hints for instance variables
    ivp: IVP
    current_time: float
    end_time: float
    step_number: int
    method_order: int
    error_constant: float


    def __init__(self, ivp, end_time, step_tol=1e-4):
        """ Initialising instance variables """
        # simple setting of input values
        self.ivp = ivp
        self.current_time = self.ivp.initial_time
        self.end_time = end_time

        self.dimension = self.ivp.get_dimension()

        # produces empty time and value meshes
        self.time_mesh = self.build_time_mesh()
        self.value_mesh = self.build_value_mesh()

        self.step_tol = step_tol
        self.method_type = MethodType.unspecified

        # setting initial values
        self.set_initial_value(self.ivp.initial_value, self.ivp.initial_time)

        # creates empty solution object for print method
        self.solution = Solution(self.time_mesh, self.value_mesh, str(self))

        self.time = [None] * 2


    def build_time_mesh(self):
        return np.zeros(self.max_mesh_size())


    def build_value_mesh(self):
        return np.zeros((self.max_mesh_size()*10, self.dimension))


    def update_value(self, step_counter, correction, call_from=MethodType.unspecified):
        if call_from == MethodType.predictor_corrector:
            self.value_mesh[step_counter] = correction


    def set_initial_value(self, initial_value, initial_time):
        self.value_mesh[0] = initial_value
        self.time_mesh[0] = initial_time


    @abstractmethod
    def pc_single_iteration(self, o_value_mesh, o_time_mesh, this_step, o_derivative_mesh=None):
        pass


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


    def __str__(self):
        return "ODE Solver Result"
