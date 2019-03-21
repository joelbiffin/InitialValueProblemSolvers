from math import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


class PendulumAnimation(object):

    def __init__(self, angles, lengths, time_mesh):
        self.time_mesh = time_mesh
        self.outputs = self.polar_to_cartesian(angles, lengths)

    @classmethod
    def from_solvers(cls, solvers, lengths, components):
        angles = [solver.value_mesh[:, c] for solver in solvers for c in components]
        time_mesh = solvers[0].time_mesh
        return cls(angles, lengths, time_mesh)


    @staticmethod
    def polar_to_cartesian(angles, lengths):
        return [np.array([
            lengths[i] * np.sin(angle), -lengths[i] * np.cos(angle)
        ]) for i, angle in enumerate(angles)]


    def show_animation(self):
        """ Credit: matplotlib examples """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, autoscale_on=False, xlim=(-16, 16), ylim=(-16, 10))
        ax.set_aspect('equal')
        ax.grid()

        line, = ax.plot([], [], 'o-', lw=2)
        time_template = 'time = %.1fs'
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

        def init():
            line.set_data([], [])
            time_text.set_text('')
            return line, time_text

        def animate(i):
            x = [0, self.outputs[0][0, i], self.outputs[0][0, i] + self.outputs[1][0, i]]
            y = [0, self.outputs[0][1, i], self.outputs[0][1, i] + self.outputs[1][1, i]]
            line.set_data(x, y)
            time_text.set_text(time_template % self.time_mesh[i])
            ani.event_source.interval = self.time_mesh[i + 1] - self.time_mesh[i]
            return line, time_text

        ani = animation.FuncAnimation(fig, animate, self.time_mesh.size,
                                      interval=1, blit=True, init_func=init)
        plt.show()

