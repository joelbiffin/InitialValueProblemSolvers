import numpy as np
import math
import scipy.optimize as opt

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
        self.set_initial_value(self.ivp.initial_value, self.ivp.initial_time)
        self.derivative_mesh[step_counter] = self.ivp.ode.function(self.value_mesh[step_counter],
                                                                   self.time_mesh[step_counter])

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


    def build_derivative_mesh(self, dimension):
        return np.zeros((self.max_mesh_size(), dimension))


    def forward_step(self, step_counter, call_from=MethodType.unspecified, u_prediction=None):
        this_step_length = self.next_step_size(step_counter)

        # calculate this iteration's current time and update corresponding
        # value in time_mesh
        self.current_time = self.time_mesh[step_counter - 1] + this_step_length
        self.time_mesh[step_counter] = self.current_time

        # calculate this iteration's approximation value
        self.value_mesh[step_counter] \
            = self.calculate_next_values(step_counter,
                                         self.next_step_size(step_counter),
                                         call_from,
                                         u_prediction)

        # calculate this iteration's derivative values
        self.derivative_mesh[step_counter] = self.calculate_next_derivative(step_counter)

    def calculate_next_derivative(self, this_step):
        return self.ivp.ode.function(self.value_mesh[this_step], self.time_mesh[this_step])


    @abstractmethod
    def calculate_next_values(self, this_step, step_size,
                              call_from=MethodType.unspecified, u_prediction=None):
        """ Calculates this step's approximation value(s) """
        pass



class AdamsBashforthSecondSolver(MultiStepSolver):

    def __init__(self, ivp, one_step_solver, end_time, step_size, precision):
        super().__init__(ivp, one_step_solver, end_time, step_size, precision, 2)
        self.method_type = MethodType.explicit


    def calculate_next_values(self, this_step, step_size, call_from=MethodType.unspecified, u_prediction=None):
        # u_{i+1} = u_i + h * ((3/2)*f(u_i, t_i) - (1/2)*f(u_{i-1}, t_{i-1}))
        u_i, t_i, f_i = (self.value_mesh[this_step - 1],
                         self.time_mesh[this_step - 1],
                         self.derivative_mesh[this_step - 1])

        # if we don't have 2 derivative values in the mesh, use forward euler
        if this_step < self.step_order:
            return self.one_step_solver.calculate_next_values(this_step, step_size)

        f_last = self.derivative_mesh[this_step - 2]
        return u_i + step_size * ((3 / 2.0)*f_i - (1 / 2.0)*f_last)


    def pc_single_iteration(self, o_value_mesh, o_time_mesh, this_step, o_derivative_mesh=None):
        u_i, t_i, f_i = (o_value_mesh[this_step - 1],
                         o_time_mesh[this_step - 1],
                         o_derivative_mesh[this_step - 1])

        # if we don't have 2 derivative values in the mesh, use forward euler
        if this_step < self.step_order:
            return self.one_step_solver.pc_single_iteration(o_value_mesh,
                                                            o_time_mesh,
                                                            this_step,
                                                            o_derivative_mesh)

        step_size = o_time_mesh[this_step] - t_i
        f_last = o_derivative_mesh[this_step - 2]

        return u_i + step_size * ((3/2.0)*f_i - (1/2.0)*f_last)



class AdamsBashforthThirdSolver(MultiStepSolver):

    def __init__(self, ivp, one_step_solver, end_time, step_size, precision):
        super().__init__(ivp, one_step_solver, end_time, step_size, precision, 3)
        self.method_type = MethodType.explicit


    def calculate_next_values(self, this_step, step_size, call_from=MethodType.unspecified, u_prediction=None):
        u_i, t_i, f_i = (self.value_mesh[this_step - 1],
                         self.time_mesh[this_step - 1],
                         self.derivative_mesh[this_step - 1])

        # if we don't have 2 derivative values in the mesh, use forward euler
        if this_step < self.step_order:
            return self.one_step_solver.calculate_next_values(this_step, step_size)

        f_last, f_llast = (self.derivative_mesh[this_step - 2],
                           self.derivative_mesh[this_step - 3])

        return u_i + (step_size / 12.0) * (23*f_i - 16*f_last + 5*f_llast)


    def pc_single_iteration(self, o_value_mesh, o_time_mesh, this_step, o_derivative_mesh=None):
        u_i, t_i, f_i = (o_value_mesh[this_step - 1],
                         o_time_mesh[this_step - 1],
                         o_derivative_mesh[this_step - 1])

        # if we don't have 2 derivative values in the mesh, use forward euler
        if this_step < self.step_order:
            print("EULERING")
            return self.one_step_solver.pc_single_iteration(o_value_mesh,
                                                            o_time_mesh,
                                                            this_step,
                                                            o_derivative_mesh)

        step_size = o_time_mesh[this_step] - t_i
        f_last, f_llast = (o_derivative_mesh[this_step - 2],
                          o_derivative_mesh[this_step - 3])

        return u_i + (step_size / 12.0) * (23*f_i - 16*f_last + 5*f_llast)






class AdamsMoultonSecondSolver(MultiStepSolver):#

    def __init__(self, ivp, one_step_solver, end_time, step_size, precision):
        super().__init__(ivp, one_step_solver, end_time, step_size, precision, 2)
        self.method_type = MethodType.implicit


    def calculate_next_values(self, this_step, step_size, call_from=MethodType.unspecified, u_prediction=None):
        # u_{i+1} = u_0 + (h/2)*(f(u_{i+1}, t_{i+1}) + f(u_i, t_i))
        u_i, t_i, f_i = (self.value_mesh[this_step - 1],
                         self.time_mesh[this_step - 1],
                         self.derivative_mesh[this_step - 1])

        # if we don't have 2 derivative values in the mesh, use one-step
        if this_step < self.step_order:
            return self.one_step_solver.calculate_next_values(this_step, step_size)

        # to find initial guess for Newton's method, we carry out forward euler
        u_guess = u_i + step_size * f_i

        # function that needs to be "solved"
        g_next = lambda u_next, derivative, t_next: \
            u_next - u_i - (step_size / 2.0) * (derivative(u_next, t_next) + f_i)

        return opt.newton(g_next, u_guess, args=(self.ivp.ode.function, self.time_mesh[this_step]))


    def pc_single_iteration(self, o_value_mesh, o_time_mesh, this_step, o_derivative_mesh=None):
        # u_{i+1} = u_0 + (h/2)*(f(u_{i+1}, t_{i+1}) + f(u_i, t_i))
        u_i, t_i, f_i = (o_value_mesh[this_step - 1],
                         o_time_mesh[this_step - 1],
                         o_derivative_mesh[this_step - 1])

        step_size = o_time_mesh[this_step] - t_i

        # if we don't have 2 derivative values in the mesh, use one-step method
        if this_step < self.step_order:
            return self.one_step_solver.calculate_next_values(this_step, step_size)

        f_next = o_derivative_mesh[this_step]

        return u_i + (step_size / 2.0) * (f_next + f_i)


class AdamsMoultonThirdSolver(MultiStepSolver):#

    def __init__(self, ivp, one_step_solver, end_time, step_size, precision):
        super().__init__(ivp, one_step_solver, end_time, step_size, precision, 3)
        self.method_type = MethodType.implicit


    def calculate_next_values(self, this_step, step_size, call_from=MethodType.unspecified, u_prediction=None):
        # u_{i+1} = u_0 + (h/2)*(f(u_{i+1}, t_{i+1}) + f(u_i, t_i))
        u_i, t_i, f_i = (self.value_mesh[this_step - 1],
                         self.time_mesh[this_step - 1],
                         self.derivative_mesh[this_step - 1])

        # if we don't have 2 derivative values in the mesh, use one-step
        if this_step < self.step_order:
            return self.one_step_solver.calculate_next_values(this_step, step_size)

        # to find initial guess for Newton's method, we carry out forward euler
        u_guess = u_i + step_size * f_i

        f_last = self.derivative_mesh[this_step - 2]

        # function that needs to be "solved"
        g_next = lambda u_next, derivative, t_next: \
            u_next - u_i - (step_size / 12.0) * (5*derivative(u_next, t_next) + 8*f_i - f_last)

        return opt.newton(g_next, u_guess, args=(self.ivp.ode.function, self.time_mesh[this_step]))


    def pc_single_iteration(self, o_value_mesh, o_time_mesh, this_step, o_derivative_mesh=None):
        # u_{i+1} = u_0 + (h/2)*(f(u_{i+1}, t_{i+1}) + f(u_i, t_i))
        u_i, t_i, f_i = (o_value_mesh[this_step - 1],
                         o_time_mesh[this_step - 1],
                         o_derivative_mesh[this_step - 1])

        step_size = o_time_mesh[this_step] - t_i

        # if we don't have 2 derivative values in the mesh, use one-step method
        if this_step < self.step_order:
            return self.one_step_solver.calculate_next_values(this_step, step_size)

        f_next, f_last = (o_derivative_mesh[this_step],
                          o_derivative_mesh[this_step - 2])

        return u_i + (step_size / 12.0) * (5*f_next + 8*f_i - f_last)



