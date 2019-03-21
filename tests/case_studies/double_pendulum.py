from math import *
import numpy as np

from src.animations import PendulumAnimation
from src.multi_step_solvers import AdamsBashforthTwoSolver, AdamsMoultonTwoSolver
from src.ode import ODE
from src.ivp import IVP
from src.one_step_solvers import ForwardEulerSolver, BackwardEulerSolver, RungeKuttaFourthSolver
from src.predictor_corrector_solvers import PredictorCorrectorSolver
from src.results import ResultsComparator



### Setting up ####################################################################################

g = 9.80665
l = [10, 5]
m = [3, 9]

def ode_system(u, t):
    denominator = l[0] * (2*m[0] + m[1] - m[1]*cos(2*u[0] - 2*u[1]))
    return np.array([
        u[2],
        u[3],
        (-g*(2*m[0] + m[1])*sin(u[0]) - m[1]*g*sin(u[0] - 2*u[2]) - 2*sin(u[0] - u[2])) / denominator,
        2*sin(u[0]-u[1])*(u[2]*u[2]*l[0]*(m[0] + m[0]) + g*(m[0]+m[1])*cos(u[0]) + u[3]*u[3]*l[1]*m[1]*cos(u[0]-u[0])) / denominator
    ])


initial_values = np.array([1, 0, 0, 0])
initial_time = 0

end_time = 100

differential_equation = ODE(ode_system)
initial_value_problem = IVP(differential_equation, initial_values, initial_time)




def adapt_step(stepped, lte, tolerance):
    update = stepped * pow((tolerance / (np.linalg.norm(lte, 2))), 0.5)

    if update > (stepped * 2):
        return 2 * stepped
    elif (2 * update) < stepped:
        return 0.5 * stepped

    return update


def local_truncation_error_estimate(prediction, correction, time_mesh=None, this_step=None):
    lte_vec = np.zeros_like(prediction)
    for i, p in enumerate(prediction):
        lte_vec[i] = fabs(p - correction[i])

    return lte_vec



### Preparing Solvers #############################################################################

def compare_methods(step_size):
    global pc_euler, pc_adams

    explicit_solver = RungeKuttaFourthSolver(initial_value_problem, end_time,
                                             step_size, step_tol=-1e-2)

    pc_euler = PredictorCorrectorSolver(
        ForwardEulerSolver(initial_value_problem, end_time, step_size, step_tol=1e-2),
        BackwardEulerSolver(initial_value_problem, end_time, step_size),
        adaptive=False, lte=local_truncation_error_estimate)

    pc_adams = PredictorCorrectorSolver(
        AdamsBashforthTwoSolver(initial_value_problem, explicit_solver, end_time, step_size),
        AdamsMoultonTwoSolver(initial_value_problem, explicit_solver, end_time, step_size),
        adaptive=False, lte=local_truncation_error_estimate)

    pc_euler.solve()
    pc_adams.solve()
    print("{} took {}s to solve."\
          .format(pc_euler, pc_euler.solve_time))
    print("{} took {}s to solve."\
          .format(pc_adams, pc_adams.solve_time))


### Comparison of results #########################################################################

this_h = 0.1
compare_methods(this_h)
comparison = ResultsComparator(
    [pc_adams, pc_adams],
    step_length=this_h, true_solution=None)
# comparison.print_result_graphs()

graphic = PendulumAnimation.from_solvers([pc_adams], l, [0, 1])
graphic.show_animation()

graphic = PendulumAnimation.from_solvers([pc_euler], l, [0, 1])
graphic.show_animation()

