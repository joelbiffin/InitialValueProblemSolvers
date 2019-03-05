import numpy as np
import matplotlib.pyplot as plt
import sys

# np.set_printoptions(threshold=sys.maxsize)
from src.solution import Solution


class ResultsComparator(object):

    approximation: Solution


    def __init__(self, approximation, true_solution):
        self.approximation = approximation
        self.true_solution = true_solution



    def print_result_graphs(self):
        """ Prints out all graphs containing true solution and our
            approximation's values.
        """
        for i in range(self.approximation.dimension):
            self.pointwise_plot(i)


    def pointwise_plot(self, system_id):
        """ Plots the approximation's values at all its mesh's points, the true
            solution is also plotted but only at these same mesh points (no
            values calculated between nodes in mesh).
                i.e. u_true(t_i) for all i, and
                     u_approx(t_i) for all i will be plotted.
        """
        # local vars for readability
        t = self.approximation.time_mesh
        pseudo_continuous_t = np.linspace(t[0], t[-1], t.size * 100)

        u_approx = self.approximation.value_mesh
        u_true = self.compute_true_values_pointwise(pseudo_continuous_t)

        # graph headings
        plt.xlabel("t")
        plt.ylabel("u_" + str(system_id) + "(t)")

        plt.plot(t, u_approx[:, system_id], color='red')
        plt.plot(pseudo_continuous_t, u_true[:, system_id], color='green')
        plt.show()


    def compute_true_values_pointwise(self, continuous_t):
        """ Evaluates the true solutions values at the approximation's time mesh
            points.
        """
        true_values = np.zeros([continuous_t.size, self.approximation.dimension])

        for i, value in enumerate(true_values):
            true_values[i] = self.true_solution(continuous_t[i])

        return true_values



