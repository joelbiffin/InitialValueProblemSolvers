import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure as figure


class ResultsComparator(object):

    def __init__(self, approximations: list, true_solution=None, step_length=None, has_lte=False):
        # solution class containing value and time meshes
        self.approximations = list(approximations)

        self.step_length = step_length

        if true_solution is not None:
            self.setup_true_solution(true_solution)
            self.setup_global_truncation_error()
        else:
            self.true_solution = None
            self.true_node_mesh = None
            self.global_truncation_error = None

        if has_lte:
            self.graph_local_truncation_errors()

    def setup_true_solution(self, true_solution):
        # function representing the true solution
        self.true_solution = true_solution
        # mesh containing the true values at all our mesh points
        self.true_node_mesh = self.compute_true_values_pointwise(
            np.trim_zeros(self.approximations[0].time_mesh, "b"))


    def setup_global_truncation_error(self):
        # initialising local_truncation_error mesh
        self.global_truncation_error \
            = [np.zeros_like(self.approximations[0])] * len(self.approximations)


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

            if u_approx.shape[0] < t.size:
                t = t[:u_approx.shape[0]]


            plt.plot(t, u_approx[:t.size, system_id], "o-", label=str(approx))

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


    def compute_global_truncation_errors(self):
        for i, approx in enumerate(self.approximations):
            self.global_truncation_error[i] \
                = (approx.value_mesh[:self.true_node_mesh.shape[0]] - self.true_node_mesh)

        return np.array(self.global_truncation_error)


    def graph_global_truncation_errors(self):
        for i in range(self.approximations[0].dimension):
            plt.figure(figsize=(12, 8))
            plt.title("Global Truncation Errors")
            plt.ylabel("Global Truncation Error")
            plt.xlabel("t")

            t = np.trim_zeros(self.approximations[0].time_mesh, "b")

            for j, error in enumerate(self.global_truncation_error):
                plt.plot(t, error[:t.size, i],
                         label="GTE in " + str(self.approximations[j]))

            plt.legend(loc="upper right")
            plt.grid()
            plt.show()


    def graph_local_truncation_errors(self):
        for i in range(self.approximations[0].dimension):
            plt.figure(figsize=(12, 8))
            plt.title("Estimated Local Truncation Error")
            plt.ylabel("LTE")
            plt.xlabel("t")

            t = np.trim_zeros(self.approximations[0].time_mesh, "b")
            lte = self.approximations[0].approx_lte[:t.size]

            plt.plot(t, lte[:t.size, i], "o-",
                     label="LTE in " + str(self.approximations[0]))

            plt.legend(loc="upper right")
            plt.grid()
            plt.show()


