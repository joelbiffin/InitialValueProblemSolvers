import numpy as np
import matplotlib.pyplot as plt

from src.solution import Solution


class ResultsComparator(object):

    approximation: Solution
    true_solution: np.ndarray


    def __init__(self, approximation, true_solution):
        self.approximation = approximation
        self.true_solution = true_solution


    def pointwise_plot(self):
        """ Plots the approximation's values at all its mesh's points, the true
            solution is also plotted but only at these same mesh points (no
            values calculated between nodes in mesh).
                i.e. u_true(t_i) for all i, and
                     u_approx(t_i) for all i will be plotted.
        """
        t = self.approximation.time_mesh
        u_approx = self.approximation.value_mesh
        print(u_approx)

        u_true = self.compute_true_values_pointwise()
        plt.plot(t, u_approx, color='red')
        plt.plot(t, u_true, color='green')
        plt.show()


    def compute_true_values_pointwise(self):
        """ Evaluates the true solutions values at the approximation's time mesh
            points.
        """
        true_values = np.zeros(self.approximation.time_mesh.size)

        for i, value in enumerate(true_values):
            true_values[i] = self.true_solution(self.approximation.time_mesh[i])

        return true_values



