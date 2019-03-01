import numpy as np


class ODE(object):
    """ Class representation of the right hand side of the initial value
        problem when in the form, u'(t) = f(u, t)
    """

    def __init__(self, function):
        self.function = function




