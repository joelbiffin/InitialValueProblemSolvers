import numpy as np
import math
import scipy.optimize as opt

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
        self.value_mesh = self.build_value_mesh(self.ivp.get_dimension())

        # creates empty solution object for print method
        self.solution = Solution(self.time_mesh, self.value_mesh, str(self))


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


    def __str__(self):
        return "ODE Solver Result"



class OneStepSolver(Solver):
    """ Class representing a numerical one-step solver (method or group of
        methods) for a given initial value problem using a fixed step-size.
    """

    # type hints for instance variables
    step_size: float
    precision: int


    def __init__(self, ivp, end_time, step_size, precision):
        """ Initialising variables same as Solver, just with constant step size.
        """
        self.step_size = step_size
        self.precision = precision
        super().__init__(ivp, end_time)

        # initial value
        self.value_mesh[0] = self.ivp.initial_value
        self.time_mesh[0] = self.ivp.initial_time


    def solve(self):
        # housekeeping
        step_counter = 0

        # loop through iterations approximating solution, storing values and
        # times used in this instance's meshes
        while self.current_time < self.end_time:
            # housekeeping variable
            step_counter += 1
            # performs operations on instance variables
            self.forward_step(step_counter)

        self.solution = Solution(self.time_mesh, self.value_mesh, str(self))


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


    def forward_step(self, step_counter):
        this_step_length = self.next_step_size(step_counter)

        # calculate this iteration's current time and update corresponding
        # value in time_mesh
        self.current_time = round(self.time_mesh[step_counter - 1] + this_step_length, self.precision)
        self.time_mesh[step_counter] = self.current_time

        # calculate this iteration's approximation value
        self.value_mesh[step_counter] \
            = self.calculate_next_values(step_counter, self.next_step_size(step_counter))


    @abstractmethod
    def calculate_next_values(self, this_step, step_size):
        """ Calculates this step's approximation value(s) """
        pass


    def __str__(self):
        return "One-Step ODE Solver"




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

        # initial values
        self.value_mesh[0] = self.ivp.initial_value
        self.time_mesh[0] = self.ivp.initial_time
        self.derivative_mesh[0] = self.ivp.ode.function(self.value_mesh[0], self.time_mesh[0])


    def solve(self):
        # housekeeping
        step_counter = 0

        # loop through iterations approximating solution, storing values and
        # times used in this instance's meshes
        while self.current_time < self.end_time:
            # housekeeping variable
            step_counter += 1

            # performs operations on instance variables
            self.forward_step(step_counter)

        self.solution = Solution(self.time_mesh, self.value_mesh, str(self))


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


    def forward_step(self, step_counter):
        this_step_length = self.next_step_size(step_counter)

        # calculate this iteration's current time and update corresponding
        # value in time_mesh
        self.current_time = round(self.time_mesh[step_counter - 1] + this_step_length, self.precision)
        self.time_mesh[step_counter] = self.current_time

        # calculate this iteration's derivative values
        self.derivative_mesh[step_counter - 1] = self.calculate_next_derivative(step_counter - 1)

        # calculate this iteration's approximation value
        self.value_mesh[step_counter] \
            = self.calculate_next_values(step_counter, self.next_step_size(step_counter))


    def calculate_next_derivative(self, this_step):
        return self.ivp.ode.function(self.value_mesh[this_step], self.time_mesh[this_step])


    @abstractmethod
    def calculate_next_values(self, this_step, step_size):
        """ Calculates this step's approximation value(s) """
        pass


    def __str__(self):
        return "Multi-Step ODE Solver"



class ForwardEulerSolver(OneStepSolver):
    def calculate_next_values(self, this_step, step_size):
        # u_{i+1} = u_i + h * f(u_i, t_i)
        u_i = self.value_mesh[this_step - 1]
        t_i = self.time_mesh[this_step - 1]
        f_i = self.ivp.ode.function(u_i, t_i)

        return u_i + step_size * f_i


    def __str__(self):
        return "Forward Euler"



class BackwardEulerSolver(OneStepSolver):
    def calculate_next_values(self, this_step, step_size):
        # u_{i+1} = u_i + h * f(u_{i+1}, t_{i+1})
        u_i = self.value_mesh[this_step - 1]
        t_i = self.time_mesh[this_step - 1]
        f_i = self.ivp.ode.function(u_i, t_i)

        # to find initial guess for Newton's method, we carry out forward euler
        u_guess = u_i + step_size * f_i

        # function that needs to be "solved"
        g_next = lambda u_next, derivative, t_next: u_next - u_i - step_size*derivative(u_next, t_next)

        return opt.newton(g_next, u_guess, args=(self.ivp.ode.function, self.time_mesh[this_step]))


    def __str__(self):
        return "Backward Euler"


class RungeKuttaFourthSolver(OneStepSolver):
    def calculate_next_values(self, this_step, step_size):
        u_i = self.value_mesh[this_step - 1]
        t_i = self.time_mesh[this_step - 1]

        k_1 = step_size * self.ivp.ode.function(u_i, t_i)
        k_2 = step_size * self.ivp.ode.function(u_i + 0.5*k_1, t_i + 0.5*step_size)
        k_3 = step_size * self.ivp.ode.function(u_i + 0.5*k_2, t_i + 0.5*step_size)
        k_4 = step_size * self.ivp.ode.function(u_i + k_3, t_i + step_size)

        return u_i + (1 / 6.0) * (k_1 + 2*k_2 + 2*k_3 + k_4)


    def __str__(self):
        return "4th Order Runge-Kutta"




class AdamsBashforthSecondSolver(MultiStepSolver):
    def __init__(self, ivp, one_step_solver, end_time, step_size, precision):
        super().__init__(ivp, one_step_solver, end_time, step_size, precision, 2)


    def calculate_next_values(self, this_step, step_size):
        # u_{i+1} = u_i + h * ((3/2)*f(u_i, t_i) - (1/2)*f(u_{i-1}, t_{i-1}))
        u_i, t_i, f_i = (self.value_mesh[this_step - 1],
                         self.time_mesh[this_step - 1],
                         self.derivative_mesh[this_step - 1])

        # if we don't have 2 derivative values in the mesh, use forward euler / runge-kutta
        if this_step < self.step_order:
            return self.one_step_solver.calculate_next_values(this_step, step_size)

        f_last = self.derivative_mesh[this_step - 2]

        return u_i + (step_size / 2.0) * (3*f_i - f_last)


    def __str__(self):
        return "2nd Order Adams-Bashforth"

