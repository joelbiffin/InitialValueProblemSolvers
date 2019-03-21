import math
import numpy as np

from src.solution import Solution
from src.solver import MethodType, Solver


class PredictorCorrectorSolver(Solver):
    """ Class representing an abstract predictor-corrector method, takes two
        solvers one explicit one implicit to solve IVP. Makes its own
        approximation
    """

    explicit_solver: Solver
    implicit_solver: Solver
    adaptive: bool


    def __str__(self):
        return str(self.explicit_solver) + "-" + str(self.implicit_solver) + " (Predictor-Corrector)"


    def __init__(self, explicit_solver, implicit_solver, adaptive=False, method=None, lte=None, two_body=False):
        # initialising solver type
        self.method_type = MethodType.predictor_corrector

        self.explicit_solver = explicit_solver
        self.implicit_solver = implicit_solver

        # boolean for whether the method uses adaptive step size or not
        self.adaptive = adaptive
        self.adapt_method = method
        self.lte = lte

        # making sure given solvers have correct type
        self.check_explicit_implicit()
        # getting initial step size
        self.step_size = self.explicit_solver.next_step_size(0)

        # checking that ivps are the same
        # TODO: make this method
        self.check_same_problem()

        self.ivp = self.explicit_solver.ivp
        self.current_time = self.explicit_solver.ivp.initial_time
        self.end_time = self.explicit_solver.end_time

        if self.explicit_solver.step_number > 1:
            self.derivative_mesh = self.build_derivative_mesh(self.ivp.dimension)
        else:
            self.derivative_mesh = None

        if self.lte is not None:
            self.approx_lte = self.explicit_solver.build_value_mesh()

        super().__init__(self.ivp, self.explicit_solver.end_time)

        self.two_body = two_body
        self.step_tol = self.explicit_solver.step_tol


    def solve(self):
        self.start_solve_time()

        # housekeeping
        step_counter = 0

        # initial value
        self.value_mesh[step_counter] = self.ivp.initial_value
        self.time_mesh[step_counter] = self.ivp.initial_time

        # loop through iterations approximating solution, storing values and
        # times used in this instance's meshes
        while self.current_time < self.end_time:

            if self.two_body:
                if self.ivp.ode.function(self.value_mesh[step_counter], self.time_mesh[step_counter]) is None:
                    break

            # housekeeping variable
            step_counter += 1
            if step_counter >= self.time_mesh.size: break
            # performs operations on instance variables
            self.forward_step(step_counter)

        self.end_solve_time()
        self.solution = Solution(self.time_mesh, self.value_mesh, str(self))


    def forward_step(self, step_counter, call_from=MethodType.unspecified, u_prediction=None):
        this_step_length = self.next_step_size(step_counter)

        # calculate this iteration's current time and update corresponding
        # value in time_mesh
        self.current_time = self.time_mesh[step_counter - 1] + this_step_length
        self.time_mesh[step_counter] = self.current_time

        # for housekeeping
        derivative = None
        lte_approx = None

        if self.derivative_mesh is None:
            prediction = self.explicit_solver.pc_single_iteration(self.value_mesh,
                                                                  self.time_mesh,
                                                                  step_counter)
            self.update_current_state(step_counter, prediction)
            correction = self.implicit_solver.pc_single_iteration(self.value_mesh,
                                                                  self.time_mesh,
                                                                  step_counter)

        else:
            prediction = self.explicit_solver.pc_single_iteration(self.value_mesh,
                                                                  self.time_mesh,
                                                                  step_counter,
                                                                  self.derivative_mesh)
            derivative = self.ivp.ode.function(prediction, self.current_time)
            self.update_current_state(step_counter, prediction, derivative)
            correction = self.implicit_solver.pc_single_iteration(self.value_mesh,
                                                                  self.time_mesh,
                                                                  step_counter,
                                                                  self.derivative_mesh)

        if self.lte is not None:
            lte_approx = self.lte(prediction, correction, self.time_mesh, step_counter)

        if self.adaptive:
            new_step = self.adapt_method(self.step_size, lte_approx, self.step_tol)

            if not(0.9*self.step_size < new_step < 1.1*self.step_size):
                self.step_size = new_step

        self.update_current_state(step_counter, correction, derivative, lte_approx)


    def update_current_state(self, step_counter, value, derivative=None, lte_approx=None):
        self.value_mesh[step_counter] = value

        if derivative is not None:
            self.derivative_mesh[step_counter] = derivative

        if self.lte is not None:
            self.approx_lte[step_counter] = lte_approx



    def max_mesh_size(self):
        return math.ceil(
            (self.end_time - self.ivp.initial_time) / self.next_step_size(0)*10) + 1


    def next_step_size(self, this_step):
        return self.step_size


    def build_derivative_mesh(self, dimension):
        return np.zeros((self.max_mesh_size(), dimension))


    def pc_single_iteration(self, o_value_mesh, o_time_mesh, this_step, o_derivative_mesh=None):
        pass


    @staticmethod
    def check_same_problem():
        return True


    def check_explicit_implicit(self):
        # TODO: change quit to exception throw
        if MethodType.explicit != self.explicit_solver.method_type:
            print("Given explicit method not explicit\n")
            quit(1)

        if MethodType.implicit != self.implicit_solver.method_type:
            print("Given implicit method not implicit")
            quit(1)


