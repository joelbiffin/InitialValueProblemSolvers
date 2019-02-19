import numpy as np


class Solution(object):

    time_mesh: np.array
    value_mesh: np.array

    def __init__(self, time_mesh, value_mesh):
        self.time_mesh = time_mesh
        self.value_mesh = value_mesh


    def __str__(self):
        return str(self.value_mesh)


