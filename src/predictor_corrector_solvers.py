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


    def __init__(self, explicit_solver, implicit_solver):
        # initialising solver type
        self.method_type = MethodType.predictor_corrector

        self.explicit_solver = explicit_solver
        self.implicit_solver = implicit_solver

        # making sure given solvers have correct type
        self.check_explicit_implicit()

        # checking that ivps are the same
        # TODO: make this method
        self.check_same_problem()

        self.ivp = self.explicit_solver.ivp
        self.current_time = self.explicit_solver.ivp.initial_time
        self.end_time = self.explicit_solver.end_time
        self.precision = self.explicit_solver.precision

        if self.explicit_solver.step_order > 1:
            self.derivative_mesh = self.build_derivative_mesh(self.ivp.dimension)
        else:
            self.derivative_mesh = None

        super().__init__(self.ivp, self.explicit_solver.end_time)
        # getting initial step size
        self.step_size = self.explicit_solver.next_step_size(0)


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

            if step_counter >= self.time_mesh.size: break

            # performs operations on instance variables
            self.forward_step(step_counter)

        self.solution = Solution(self.time_mesh, self.value_mesh)



    def forward_step(self, step_counter, call_from=MethodType.unspecified, u_prediction=None):
        this_step_length = self.next_step_size(step_counter)

        # calculate this iteration's current time and update corresponding
        # value in time_mesh
        self.current_time = (self.time_mesh[step_counter - 1] + this_step_length)
        self.time_mesh[step_counter] = self.current_time

        # for housekeeping
        derivative = None

        if self.derivative_mesh is None:
            prediction = self.explicit_solver.pc_single_iteration(self.value_mesh,
                                                                  self.time_mesh,
                                                                  step_counter)
            self.update_current_state(step_counter, prediction)
            correction = self.implicit_solver.pc_single_iteration(self.value_mesh,
                                                                  self.time_mesh,
                                                                  step_counter,
                                                                  prediction)
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
            print("prediction:\t", prediction)
            print("derivative:\t", derivative)
            print("correction:\t", correction)

        self.update_current_state(step_counter, correction, derivative)


    def update_current_state(self, step_counter, value, derivative=None):
        self.value_mesh[step_counter] = value

        if not(derivative is None):
            self.derivative_mesh[step_counter] = derivative


    def max_mesh_size(self):
        return math.ceil(
            (self.end_time - self.ivp.initial_time) / self.next_step_size(0)) + 1


    def next_step_size(self, this_step):
        return self.explicit_solver.next_step_size(0)


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


