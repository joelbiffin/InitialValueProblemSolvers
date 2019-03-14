import math as m
import numpy as np

from src.multi_step_solvers import AdamsBashforthTwoSolver
from src.ode import ODE
from src.ivp import IVP
from src.one_step_solvers import ForwardEulerSolver, BackwardEulerSolver, RungeKuttaFourthSolver
from src.predictor_corrector_solvers import PredictorCorrectorSolver
from src.results import ResultsComparator


"""
Case study 1: Harmonic Oscillator (1)

    > Takes 2nd order scalar ordinary differential equation with
      constant coefficients and known solution.
      
    > Initial Value Problem:
        x''(t) + 2x'(t) + 2x(t) = 30cos(2t),
        x(0) = 0, x'(0) = 10
        
    > Reduction to System of 1st Order IVPs:
      Let u_1 = x(t), u_2 = x'(t), then we arrive at the coupled system,
        u_1 = u_2
        u_2 = 30cos(2t) - 2u_1 - 2u_2,
      with initial conditions,
        u_1(0) = 0
        u_2(0) = 10
    
    > True Solution:
        x(t) = e^(-t) (3cos(t) + sin(t)) + 6sin(2t) - 3cos(2t)

"""

### Setting up ####################################################################################

true_solution = lambda t: np.array([m.exp(-t)*(3*m.cos(t) + m.sin(t)) \
                          + 6*m.sin(2*t) - 3*m.cos(2*t), 0])

ode_system = lambda u, t: np.array([
    u[1],
    30*m.cos(2*t) - 2*u[0] - 2*u[1]
])

initial_values = np.array([0, 10])
initial_time = 0

end_time = 10
#step_le = 0.25

differential_equation = ODE(ode_system)
initial_value_problem = IVP(differential_equation, initial_values, initial_time)


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
        ForwardEulerSolver(initial_value_problem,end_time, step_size, step_tol=1),
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

compare_one_step_methods(.1)
comparison = ResultsComparator([forward_euler, backward_euler, runge_kutta,
                                predictor_corrector, adams_bashforth],
                               true_solution=true_solution)
comparison.print_result_graphs()




