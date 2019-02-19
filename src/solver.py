import numpy as np
import math

from abc import abstractmethod, ABCMeta
from src.ivp import IVP
from src.solution import Solution


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


    def __init__(self, ivp, end_time):
        """ Initialising instance variables """
        # simple setting of input values
        self.ivp = ivp
        self.current_time = self.ivp.initial_time
        self.end_time = end_time

        # produces empty time and value meshes
        self.time_mesh = self.build_time_mesh()
        # TODO: create variable value mesh size for vector equations
        self.value_mesh = self.build_value_mesh(2)

        # creates empty solution object for print method
        self.solution = Solution(self.time_mesh, self.value_mesh)


    def build_time_mesh(self):
        return np.zeros(self.max_mesh_size())


    def build_value_mesh(self, dimension):
        return np.zeros([self.max_mesh_size(), dimension])


    def print_solution(self):
        print(self.solution)


    @abstractmethod
    def solve(self):
        """ Running approximation up until end_time """
        pass


    @abstractmethod
    def forward_step(self, step_counter):
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
    def next_step_size(self):
        """ Return the step size for the next iteration of the method. """
        pass



class OneStepSolver(Solver):
    """ Class representing a numerical one-step solver (method or group of
        methods) for a given initial value problem using a fixed step-size.
    """

    # type hints for instance variables
    step_size: float


    def __init__(self, ivp, end_time, step_size):
        """ Initialising variables same as Solver, just with constant step size.
        """
        self.step_size = step_size
        super().__init__(ivp, end_time)


    def solve(self):
        # housekeeping
        step_counter = int(0)
        """print("u_0:")
        print(self.ivp.initial_value.shape)
        print()

        print("mesh:")
        print(self.value_mesh)
        input("HELLLOO")
        """

        # initial value
        self.value_mesh[step_counter] = self.ivp.initial_value
        self.time_mesh[step_counter] = self.ivp.initial_time

        """
        input("HELLLOO")
        quit()
        """

        # loop through iterations approximating solution, storing values and
        # times used in this instance's meshes
        while self.current_time <= self.end_time:
            print("HH")
            # housekeeping variable
            step_counter += 1
            # performs operations on instance variables
            self.forward_step(step_counter)

        self.solution = Solution(self.time_mesh, self.value_mesh)


    def next_step_size(self, this_step):
        return self.step_size


    def max_mesh_size(self):
        """ Provides the mesh required for approximation, given pre-defined
            step_size and start and end times
        """
        # note that we use the literal 0 WLOG since we have a fixed step-size
        # in one step methods
        return math.ceil(
            (self.end_time - self.ivp.initial_time + 1) / self.next_step_size(0))


    def forward_step(self, step_counter):
        this_step_length = self.next_step_size(step_counter)

        # calculate this iteration's current time and update corresponding
        # value in time_mesh
        self.current_time = self.time_mesh[step_counter - 1] + this_step_length
        self.time_mesh[step_counter] = self.current_time

        # calculate this iteration's approximation value
        self.value_mesh[step_counter] \
            = self.calculate_next_values(step_counter, self.next_step_size(step_counter))


    @abstractmethod
    def calculate_next_values(self, this_step, step_size):
        """ Calculates this step's approximation value(s) """
        pass



class ForwardEulerSolver(OneStepSolver):
    def calculate_next_values(self, this_step, step_size):
        # u_{i+1} = u_i + h * f(u_i, t_i)
        u_i = self.value_mesh[this_step - 1]
        t_i = self.time_mesh[this_step - 1]
        print("u_i", u_i)
        print("t_i", t_i)

        f_i = self.ivp.ode.compute_derivative(u_i, t_i)

        # print("(APPROX) u(", t_i , ") ~ ", u_i + step_size * f_i)

        # TODO: ambigious how these operations happen with numpy arrays
        return u_i + step_size * f_i



class BackwardEulerSolver(OneStepSolver):
    pass



class MultiStepSolver(Solver):
    pass



class MultiStepFixedWidthSolver(MultiStepSolver):
    pass



class MultiStepAdaptiveWidthSolver(MultiStepSolver):
    pass




