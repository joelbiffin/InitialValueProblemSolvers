import numpy as np
import matplotlib.pyplot as plt
import sys


from src.solution import Solution


class ResultsComparator(object):

    approximation: Solution


    def __init__(self, approximation, true_solution):
        # function representing the true solution
        self.true_solution = true_solution

        # solution class containing value and time meshes
        self.approximation = approximation

        # mesh containing the true values at all our mesh points
        self.true_node_mesh = self.compute_true_values_pointwise(self.approximation.time_mesh)

        # initialising local_truncation_error mesh
        self.local_truncation_error = np.zeros((
            self.approximation.time_mesh.size, self.approximation.dimension
        ))


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
        plt.title(self.approximation.method_title)
        plt.plot(t, u_approx[:, system_id], color="red", label="Approximation")
        plt.plot(pseudo_continuous_t, u_true[:, system_id],
                 color="green", label="True Solution")
        plt.legend(loc="upper right")
        plt.grid()
        plt.show()


    def compute_true_values_pointwise(self, t):
        """ Evaluates the true solutions values at the approximation's time mesh
            points.
        """
        true_values = np.zeros([t.size, self.approximation.dimension])

        for i, value in enumerate(true_values):
            true_values[i] = self.true_solution(t[i])

        return true_values


    def compute_local_truncation_errors(self):
        self.local_truncation_error = self.approximation.value_mesh - self.true_node_mesh
        return self.local_truncation_error


    def graph_local_truncation_errors(self):
        for i in range(self.approximation.dimension):
            plt.title("LTE in "+ str(self.approximation.method_title))
            plt.ylabel("Local Truncation Error")
            plt.xlabel("t")
            plt.plot(self.approximation.time_mesh, self.local_truncation_error[:, i], color="red", label="LTE")
            plt.legend(loc="upper right")
            plt.grid()
            plt.show()


