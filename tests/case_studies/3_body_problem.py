import numpy as np

from math import *
from src.multi_step_solvers import AdamsBashforthTwoSolver, AdamsMoultonTwoSolver
from src.ode import ODE
from src.ivp import IVP
from src.one_step_solvers import ForwardEulerSolver, BackwardEulerSolver, RungeKuttaFourthSolver
from src.predictor_corrector_solvers import PredictorCorrectorSolver
from src.results import ResultsComparator

"""
Case study 2: Gravitational 2-body Problem

    > Takes a coupled 2nd order system of IVPs in reduced form

    > Initial Value Problem:
        Force of j on i:  r''_ij(t) = (-G m_2)/(|r_1-r_2|^3) * r

    > Reduction to System of 1st Order IVPs:
      Let u_1 = r_12(t), u_2 = r'_12(t), v_1 = r_21(t), v_2 = r'_21(t),
      then we arrive at the coupled system,
        u_1 = u_2
        u_2 = (-G m_2)/(|u_1 - v_1|^3) (u_1-v_1)
        v_1 = v_2
        v_2 = (G m_1).(|u_1 - v_1|^3) (v_1 - u_1,

      with initial conditions,
        u_1(0) = 0
        u_2(0) = 10

        m_1 is mass of the earth
        m_2 is mass of the moon


        m_1 = 10
        m_2 = 12
        u_1(0)=20
        u_2(0)=0
        v_1(


"""

### Setting up ####################################################################################

m_0 = 6
m_1 = 5
m_2 = 3

u_initial = np.array([
    1,
    -1,
    0,
    0,

    -1,
    1,
    0,
    0,

    -1,
    0,
    0,
    0
])


def ode_system(u, t):
    direction10 = u[4:6] - u[0:2]
    direction21 = u[8:10] - u[4:6]
    direction02 = u[0:2] - u[8:10]
    r10_cubed = pow(np.linalg.norm(direction10, 2), 3)
    r21_cubed = pow(np.linalg.norm(direction21, 2), 3)
    r02_cubed = pow(np.linalg.norm(direction02, 2), 3)


    if (r10_cubed < 10e-10) or (r21_cubed < 10e-10) or (r02_cubed < 10e-10):
        return None

    return np.array([
        u[2],
        u[3],
        (m_1 * direction10[0]) / r10_cubed - (m_2 * direction02[0]) / r02_cubed,
        (m_1 * direction10[1]) / r10_cubed - (m_2 * direction02[1]) / r02_cubed,

        u[6],
        u[7],
        (m_2 * direction21[0]) / r21_cubed - (m_0 * direction10[0]) / r10_cubed,
        (m_2 * direction21[1]) / r21_cubed - (m_0 * direction10[1]) / r10_cubed,

        u[10],
        u[11],
        (m_0 * direction02[0]) / r02_cubed - (m_1 * direction21[0]) / r21_cubed,
        (m_0 * direction02[1]) / r02_cubed - (m_1 * direction21[1]) / r21_cubed
    ])


"""
ode_system = lambda u, t: np.array([
    u[6],
    u[7],
    u[8],
    (-1*(G*m_2) / (r*r*r))*(u[0]-u[6]),
    (-1*(G*m_2) / (r*r*r))*(u[1]-u[7])
])

"""
initial_time = 0
end_time = 100

differential_equation = ODE(ode_system)
initial_value_problem = IVP(differential_equation, u_initial, initial_time)


### Preparing Solvers #############################################################################


def adapt_step(stepped, lte, tolerance):
    update = stepped * pow((tolerance / (np.linalg.norm(lte, 2))), (1.0 / 3))

    if update > (stepped * 2):
        return 2 * stepped
    elif (2 * update) < stepped:
        return 0.5 * stepped

    return update


def local_truncation_error_estimate(prediction, correction, time_mesh, this_step):
    lte_vec = np.zeros_like(prediction)
    for i, p in enumerate(prediction):
        lte_vec[i] = fabs(p - correction[i])

    steps = [time_mesh[this_step] - time_mesh[this_step - 1],
             time_mesh[this_step - 1] - time_mesh[this_step - 2]]

    return lte_vec / (3 + (steps[1] / steps[0]))


def compare_one_step_methods(step_size):
    global runge_kutta, predictor_corrector

    runge_kutta = RungeKuttaFourthSolver(initial_value_problem, end_time, step_size, two_body=True)
    # runge_kutta.solve()

    one_step = RungeKuttaFourthSolver(initial_value_problem, end_time, step_size, step_tol=1e-2)

    predictor_corrector = PredictorCorrectorSolver(
        AdamsBashforthTwoSolver(initial_value_problem, one_step, end_time, step_size),
        AdamsMoultonTwoSolver(initial_value_problem, one_step, end_time, step_size),
        adaptive=True, method=adapt_step, lte=local_truncation_error_estimate, two_body=True)
    predictor_corrector.solve()


### Comparison of results #########################################################################

"""compare_one_step_methods(0.5)
comparison = ResultsComparator([forward_euler, backward_euler, runge_kutta],
                               true_solution=true_solution)
comparison.print_result_graphs()


compare_one_step_methods(0.25)
comparison = ResultsComparator([forward_euler, backward_euler, runge_kutta],
                               true_solution=true_solution)
comparison.print_result_graphs()
"""

compare_one_step_methods(1e-2)

comparison = ResultsComparator([predictor_corrector], true_solution=None)
comparison.print_result_graphs()

# comparison_2 = ResultsComparator([runge_kutta], true_solution=None)
# comparison_2.print_result_graphs()
# comparison.graph_local_truncation_errors()
# comparison = ResultsComparator([runge_kutta], true_solution=None)
# comparison.print_result_graphs()

predictor_corrector.solution.write_to_csv("../../outputs/3_body.csv", [0, 1, 4, 5, 8, 9])



