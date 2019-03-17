import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure as figure


class ResultsComparator(object):

    approximations: list


    def __init__(self, approximations, true_solution=None, step_length=None):
        # solution class containing value and time meshes
        self.approximations = approximations

        self.step_length = step_length
        
        if true_solution is not None:
            self.setup_true_solution(true_solution)
            self.setup_local_truncation_error()
        else:
            self.true_solution = None
            self.true_node_mesh = None
            self.local_truncation_error = None


    def setup_true_solution(self, true_solution):
        # function representing the true solution
        self.true_solution = true_solution
        # mesh containing the true values at all our mesh points
        self.true_node_mesh = self.compute_true_values_pointwise(self.approximations[0].time_mesh)


    def setup_local_truncation_error(self):
        # initialising local_truncation_error mesh
        self.local_truncation_error = np.zeros((
            self.approximations[0].time_mesh.size, self.approximations[0].dimension
        ))


    def print_result_graphs(self):
        """ Prints out all graphs containing true solution and our
            approximation's values.
        """
        for j in range(self.approximations[0].dimension):
            self.pointwise_plot(j)


    def pointwise_plot(self, system_id):
        """ Plots the approximation's values at all its mesh's points, the true
            solution is also plotted but only at these same mesh points (no
            values calculated between nodes in mesh).
                i.e. u_true(t_i) for all i, and
                     u_approx(t_i) for all i will be plotted.
        """

        t = np.trim_zeros(self.approximations[0].time_mesh, "b")
        plt.figure(figsize=(12, 8))

        if self.step_length is None:
            plt.title("Comparison of ODE Methods")
        else:
            plt.title("Comparison of ODE Methods h=("+ str(self.step_length)+ ")")

        for approx in self.approximations:
            u_approx = approx.value_mesh[:t.size]
            plt.plot(t, u_approx[:, system_id], "o-", label=str(approx))

        if self.true_solution is not None:
            pseudo_continuous_t = np.linspace(t[0], t[-1], t.size*50)
            u_true = self.compute_true_values_pointwise(pseudo_continuous_t)
            plt.plot(pseudo_continuous_t, u_true[:, system_id], label="True Solution")

        plt.legend(loc="upper right")
        plt.grid()
        plt.show()


    def compute_true_values_pointwise(self, t):
        """ Evaluates the true solutions values at the approximation's time mesh
            points.
        """
        true_values = np.zeros([t.size, self.approximations[0].dimension])

        for i, value in enumerate(true_values):
            true_values[i] = self.true_solution(t[i])

        return true_values


    def compute_local_truncation_errors(self):
        self.local_truncation_error = self.approximations[0].value_mesh - self.true_node_mesh
        return self.local_truncation_error


    def graph_local_truncation_errors(self):
        for i in range(self.approximations[0].dimension):
            plt.title("LTE in "+ str(self.approximations[0]))
            plt.ylabel("Local Truncation Error")
            plt.xlabel("t")
            plt.plot(self.approximations[0].time_mesh, self.local_truncation_error[:, i], color="red", label="LTE")
            plt.legend(loc="upper right")
            plt.grid()
            plt.show()




