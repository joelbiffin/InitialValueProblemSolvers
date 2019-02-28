import numpy as np
import matplotlib.pyplot as plt

from src.solution import Solution


class ResultsComparator(object):

    approximation: Solution


    def __init__(self, approximation, true_solution):
        self.approximation = approximation
        self.true_solution = true_solution

        #print(self.approximation.)


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
        u_approx = self.approximation.value_mesh
        u_true = self.compute_true_values_pointwise()

        # graph headings
        plt.xlabel("t")
        plt.ylabel("u_" + str(system_id) + "(t)")

        # plotting data on graph
        plt.plot(t, u_approx[:, system_id], color='red')
        plt.plot(t, u_true[:, system_id], color='green')
        plt.show()


    def compute_true_values_pointwise(self):
        """ Evaluates the true solutions values at the approximation's time mesh
            points.
        """
        true_values = np.zeros([self.approximation.time_mesh.size,
                               self.approximation.dimension])

        for i, value in enumerate(true_values):
            true_values[i] = self.true_solution(self.approximation.time_mesh[i])

        return true_values



