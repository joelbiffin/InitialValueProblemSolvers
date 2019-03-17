import math
import numpy as np

from src.multi_step_solvers import AdamsBashforthTwoSolver
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
G = 6.67408e-11
m_1 = 5.972e24
m_2 = 7.347e22
r_earth = 6.371e6
r_moon = 1.737e6

u_initial = np.array([0, 0, 0, 0, 0, 0, r_earth+r_moon+370300e3, 0, 0, 0, 24*3.683e3, 0])



def ode_system(u, t):
    r = np.linalg.norm(u[0:3] - u[6:9], 2)
    print(r)
    constant_1 = -1*(G*m_2) / (math.pow(r, 3))
    constant_2 = -1*(G*m_1) / (math.pow(r, 3))
    return np.array([
        u[6],
        u[7],
        u[8],
        constant_1 * (u[0] - u[6]),
        constant_1 * (u[1] - u[7]),
        constant_1 * (u[2] - u[8]),
        u[0],
        u[1],
        u[2],
        constant_2 * (u[6] - u[0]),
        constant_2 * (u[7] - u[1]),
        constant_2 * (u[8] - u[2])
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

def compare_one_step_methods(step_size):
    global forward_euler, backward_euler, runge_kutta, predictor_corrector, adams_bashforth

    forward_euler = ForwardEulerSolver(initial_value_problem, end_time, step_size)
    backward_euler = BackwardEulerSolver(initial_value_problem, end_time, step_size)
    runge_kutta = RungeKuttaFourthSolver(initial_value_problem, end_time, step_size)
    adams_bashforth = AdamsBashforthTwoSolver(
        initial_value_problem,
        RungeKuttaFourthSolver(initial_value_problem, end_time, step_size),
        end_time, step_size)

    predictor_corrector = PredictorCorrectorSolver(
        ForwardEulerSolver(initial_value_problem, end_time, step_size, step_tol=1e-1),
        BackwardEulerSolver(initial_value_problem, end_time, step_size),
        adaptive=False)

    forward_euler.solve()
    backward_euler.solve()
    runge_kutta.solve()
    adams_bashforth.solve()
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

compare_one_step_methods(.2)
#comparison = ResultsComparator([forward_euler, backward_euler, runge_kutta,
#                                predictor_corrector, adams_bashforth],
#                               true_solution=None)

comparison = ResultsComparator([runge_kutta], true_solution=None)
comparison.print_result_graphs()




