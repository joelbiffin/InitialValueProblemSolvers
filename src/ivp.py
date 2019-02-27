import numpy as np
from src.ode import ODE


class IVP(object):
    """ Class representing an abstract initial value problem, consisting
        of initial conditions and a corresponding right hand side to the
        (vector) ODE.
            u'(t) = f(u, t),
            u(t_0) = u_0
    """

    ode: ODE
    initial_value: np.ndarray
    initial_time: float
    dimension: int


    def __init__(self, ode, initial_value, initial_time):
        """ Setting instance variables """
        self.ode = ode
        self.initial_value = initial_value
        self.initial_time = initial_time
        self.dimension = self.initial_value.size


    def get_dimension(self):
        return self.dimension

