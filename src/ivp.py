import numpy as np
from src.ode import ODE


class IVP(object):
    """ Class representing an abstract initial value problem, consisting
        of initial conditions and a corresponding right hand side to the
        (vector) ODE.
            u'(t) = f(u, t),
            u(t_0) = u_0
    """

    def __init__(self, ode, initial_value, initial_time):
        """ Setting instance variables """
        self.ode = ode
        self.initial_value = initial_value
        self.initial_time = initial_time





def f(u, t):
    return u


de = ODE(f)

u_0 = np.array([1])
t_0 = 0

problem = IVP(de, u_0, t_0)

print(problem.ode.compute_derivative(2, 0))
