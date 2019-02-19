import numpy as np


class ODE(object):
    """ Class representation of the right hand side of the initial value
        problem when in the form, u'(t) = f(u, t)
    """

    function: np.array

    def __init__(self, function):
        self.function = function


    def compute_derivative(self, value, time):
        return self.function(value, time)


    def get_dimension(self):
        return self.function(0, 0)

