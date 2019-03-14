import math
import scipy.optimize as opt

from abc import abstractmethod
from src.solution import Solution
from src.solver import MethodType, Solver


class OneStepSolver(Solver):
    """ Class representing a numerical one-step solver (method or group of
        methods) for a given initial value problem using a fixed step-size.
    """

    # type hints for instance variables
    step_size: float
    step_number: int


    def __init__(self, ivp, end_time, step_size, step_tol=1e-4):
        """ Initialising variables same as Solver, just with constant step size.
        """
        self.step_size = step_size
        self.step_number = 1
        super().__init__(ivp, end_time, step_tol=step_tol)


    def solve(self):
        # housekeeping
        step_counter = 0

        # initial value
        self.value_mesh[step_counter] = self.ivp.initial_value
        self.time_mesh[step_counter] = self.ivp.initial_time

        # loop through iterations approximating solution, storing values and
        # times used in this instance's meshes
        while self.current_time < self.end_time:
            # housekeeping variable
            step_counter += 1
            # checking we aren't out of bounds of time mesh
            if step_counter >= self.time_mesh.size: break
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


    def forward_step(self, step_counter, call_from=MethodType.unspecified, u_prediction=None):
        this_step_length = self.next_step_size(step_counter)

        # calculate this iteration's current time and update corresponding
        # value in time_mesh
        self.current_time = self.time_mesh[step_counter - 1] + this_step_length
        self.time_mesh[step_counter] = self.current_time

        # calculate this iteration's approximation value
        self.value_mesh[step_counter] \
            = self.calculate_next_values(step_counter, self.next_step_size(step_counter))

        return self.value_mesh[step_counter]



    @abstractmethod
    def calculate_next_values(self, this_step, step_size,
                              call_from=MethodType.unspecified, u_prediction=None):
        """ Calculates this step's approximation value(s) """
        pass




class ForwardEulerSolver(OneStepSolver):

    def __str__(self):
        return "Forward Euler"


    def __init__(self, ivp, end_time, step_size, step_tol=1e-4):
        super().__init__(ivp, end_time, step_size, step_tol=step_tol)
        self.method_type = MethodType.explicit
        self.method_order = 1
        self.error_constant = 0.5

    def calculate_next_values(self, this_step, step_size,
                              call_from=MethodType.unspecified, u_prediction=None):
        # u_{i+1} = u_i + h * f(u_i, t_i)
        u_i = self.value_mesh[this_step - 1]
        t_i = self.time_mesh[this_step - 1]
        f_i = self.ivp.ode.function(u_i, t_i)

        return u_i + step_size * f_i


    def pc_single_iteration(self, o_value_mesh, o_time_mesh, this_step, o_derivative_mesh=None):
        u_i = o_value_mesh[this_step - 1]
        t_i = o_time_mesh[this_step - 1]
        f_i = self.ivp.ode.function(u_i, t_i)
        step_size = o_time_mesh[this_step] - t_i

        return u_i + step_size * f_i



class BackwardEulerSolver(OneStepSolver):

    def __str__(self):
        return "Backward Euler"

    def __init__(self, ivp, end_time, step_size, step_tol=1e-4):
        super().__init__(ivp, end_time, step_size, step_tol=step_tol)
        self.method_type = MethodType.implicit
        self.method_order = 1
        self.error_constant = -0.5


    def calculate_next_values(self, this_step, step_size,
                              call_from=MethodType.unspecified, u_prediction=None):
        u_i = self.value_mesh[this_step - 1]

        # if called from predictor-corrector method, use given guess value
        # instead of Newton's method
        if call_from == MethodType.predictor_corrector:
            t_next = self.time_mesh[this_step]
            f_next = self.ivp.ode.function(u_prediction, t_next)
            return u_i + step_size * f_next

        # u_{i+1} = u_i + h * f(u_{i+1}, t_{i+1})
        t_i = self.time_mesh[this_step - 1]
        f_i = self.ivp.ode.function(u_i, t_i)

        # to find initial guess for Newton's method, we carry out forward euler
        u_guess = u_i + step_size * f_i
        # function that needs to be "solved"
        g_next = lambda u_next, derivative, t_next: u_next - u_i - step_size * derivative(u_next, t_next)

        return opt.fsolve(g_next, u_guess, args=(self.ivp.ode.function, self.time_mesh[this_step]))


    def pc_single_iteration(self, o_value_mesh, o_time_mesh, this_step, o_derivative_mesh=None):
        u_i = o_value_mesh[this_step - 1]
        t_next = o_time_mesh[this_step]

        step_size = t_next - o_time_mesh[this_step - 1]

        # note that the value given by our predictor is, for now, stored as the value in
        # o_value_mesh[this_step]
        f_next = self.ivp.ode.function(o_value_mesh[this_step], t_next)

        return u_i + step_size * f_next


class RungeKuttaFourthSolver(OneStepSolver):

    def __str__(self):
        return "Runge-Kutta 4-Stage"

    def __init__(self, ivp, end_time, step_size, step_tol=1e-4):
        super().__init__(ivp, end_time, step_size, step_tol=step_tol)
        self.method_type = MethodType.explicit


    def calculate_next_values(self, this_step, step_size,
                              call_from=MethodType.unspecified, u_expected=None):
        u_i = self.value_mesh[this_step - 1]
        t_i = self.time_mesh[this_step - 1]

        k_1 = step_size * self.ivp.ode.function(u_i, t_i)
        k_2 = step_size * self.ivp.ode.function(u_i + 0.5*k_1, t_i + 0.5*step_size)
        k_3 = step_size * self.ivp.ode.function(u_i + 0.5*k_2, t_i + 0.5*step_size)
        k_4 = step_size * self.ivp.ode.function(u_i + k_3, t_i + step_size)

        return u_i + (1 / 6.0) * (k_1 + 2*k_2 + 2*k_3 + k_4)


    def pc_single_iteration(self, o_value_mesh, o_time_mesh, this_step, o_derivative_mesh=None):
        u_i = o_value_mesh[this_step - 1]
        t_i = o_time_mesh[this_step - 1]
        step_size = o_time_mesh[this_step] - t_i

        k_1 = step_size * self.ivp.ode.function(u_i, t_i)
        k_2 = step_size * self.ivp.ode.function(u_i + 0.5*k_1, t_i + 0.5*step_size)
        k_3 = step_size * self.ivp.ode.function(u_i + 0.5*k_2, t_i + 0.5*step_size)
        k_4 = step_size * self.ivp.ode.function(u_i + k_3, t_i + step_size)

        return u_i + (1 / 6.0) * (k_1 + 2*k_2 + 2*k_3 + k_4)



