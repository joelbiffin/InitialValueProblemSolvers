import numpy as np
import math

from abc import abstractmethod
from src.solution import Solution
from src.solver import MethodType, Solver


# TODO: refactor inheritance to optimise code re-use
# TODO: design decision regarding variable step-length methods



class MultiStepSolver(Solver):
    """ Class representing abstract multi-step ode solver. """


    def __init__(self, ivp, one_step_solver, end_time, step_size, precision, step_order):
        """ Initialising variables same as Solver, just with constant step size.
        """
        self.step_size = step_size
        self.precision = precision

        # contains how many steps are considered for the ith iteration's approximation
        self.step_order = step_order

        # contains an instance of a one-step solver for computation of the first few steps
        self.one_step_solver = one_step_solver

        super().__init__(ivp, end_time)

        # builds mesh to contain derivatives evaluated at each step
        self.derivative_mesh = self.build_derivative_mesh(self.ivp.get_dimension())


    def solve(self):
        # housekeeping
        step_counter = 0

        # initial value
        self.value_mesh[step_counter] = self.ivp.initial_value
        self.time_mesh[step_counter] = self.ivp.initial_time
        self.derivative_mesh[step_counter] = self.ivp.ode.function(
            self.value_mesh[step_counter],
            self.time_mesh[step_counter]
        )

        # loop through iterations approximating solution, storing values and
        # times used in this instance's meshes
        while self.current_time < self.end_time:
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
            (self.end_time - self.ivp.initial_time) / self.next_step_size(0)) + 1


    def build_derivative_mesh(self, dimension):
        return np.zeros((self.max_mesh_size(), dimension))


    def forward_step(self, step_counter, call_from=MethodType.unspecified, u_prediction=None):
        this_step_length = self.next_step_size(step_counter)

        # calculate this iteration's current time and update corresponding
        # value in time_mesh
        self.current_time = round(self.time_mesh[step_counter - 1] + this_step_length, self.precision)
        self.time_mesh[step_counter] = self.current_time

        # calculate this iteration's derivative values
        self.derivative_mesh[step_counter] = self.calculate_next_derivative(step_counter)

        # calculate this iteration's approximation value
        self.value_mesh[step_counter] \
            = self.calculate_next_values(step_counter,
                                         self.next_step_size(step_counter),
                                         call_from,
                                         u_prediction)


    def calculate_next_derivative(self, this_step):
        return self.ivp.ode.function(self.value_mesh[this_step], self.time_mesh[this_step])


    @abstractmethod
    def calculate_next_values(self, this_step, step_size,
                              call_from=MethodType.unspecified, u_prediction=None):
        """ Calculates this step's approximation value(s) """
        pass



class AdamsBashforthSecondSolver(MultiStepSolver):

    def __init__(self, ivp, one_step_solver, end_time, step_size, precision):
        self.method_type = MethodType.explicit
        super().__init__(ivp, one_step_solver, end_time, step_size, precision, 2)


    def calculate_next_values(self, this_step, step_size, call_from=MethodType.unspecified):
        # u_{i+1} = u_i + h * ((3/2)*f(u_i, t_i) - (1/2)*f(u_{i-1}, t_{i-1}))
        u_i, t_i, f_i = (self.value_mesh[this_step - 1],
                         self.time_mesh[this_step - 1],
                         self.derivative_mesh[this_step - 1])

        # if we don't have 2 derivative values in the mesh, use forward euler
        if this_step < self.step_order:
            return self.one_step_solver.calculate_next_values(this_step, step_size)

        f_last = self.derivative_mesh[this_step - 2]
        return u_i + step_size * ((3 / 2.0)*f_i - (1 / 2.0)*f_last)

