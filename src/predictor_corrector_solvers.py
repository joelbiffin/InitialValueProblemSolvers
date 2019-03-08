import math

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
        self.check_same_problem()

        self.ivp = self.explicit_solver.ivp
        self.current_time = self.explicit_solver.ivp.initial_time
        self.end_time = self.explicit_solver.end_time
        print(self.current_time)

        # getting initial step size
        self.step_size = self.explicit_solver.next_step_size(0)

        super().__init__(self.ivp, self.explicit_solver.end_time)


    def solve(self):
        # housekeeping
        step_counter = 0

        # initial value
        self.value_mesh[step_counter] = self.ivp.initial_value
        self.time_mesh[step_counter] = self.ivp.initial_time

        print(self.current_time, self.end_time, sep=' ')

        # loop through iterations approximating solution, storing values and
        # times used in this instance's meshes
        while self.current_time < self.end_time:
            # housekeeping variable
            step_counter += 1
            # performs operations on instance variables
            self.forward_step(step_counter)
            print(self.time_mesh)

        self.solution = Solution(self.time_mesh, self.value_mesh)



    def forward_step(self, step_counter, call_from=MethodType.unspecified, u_prediction=None):
        this_step_length = self.next_step_size(step_counter)

        # calculate this iteration's current time and update corresponding
        # value in time_mesh
        self.current_time = self.time_mesh[step_counter - 1] + this_step_length
        self.time_mesh[step_counter] = self.current_time

        prediction = self.explicit_solver.forward_step(step_counter)
        correction = self.implicit_solver.forward_step(step_counter,
                                                       call_from=MethodType.predictor_corrector,
                                                       u_prediction=prediction)
        self.value_mesh[step_counter] = correction


    def max_mesh_size(self):
        return math.ceil(
            (self.end_time - self.ivp.initial_time) / self.next_step_size(0)) + 1


    def next_step_size(self, this_step):
        return self.step_size


    @staticmethod
    def check_same_problem():
        return True


    def check_explicit_implicit(self):
        # TODO: change quit to exception throw
        if self.explicit_solver.method_type != MethodType.explicit:
            print("Given explicit method not explicit\n")
            quit(1)

        if self.implicit_solver.method_type != MethodType.implicit:
            print("Given implicit method not implicit")
            quit(1)


