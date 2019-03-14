import numpy as np


class Solution(object):

    time_mesh: np.array
    value_mesh: np.array
    dimension: int


    def __init__(self, time_mesh, value_mesh, method_title="Unknown"):
        self.time_mesh = time_mesh
        self.value_mesh = value_mesh
        self.dimension = self.value_mesh.shape[1]
        self.method_title = method_title


    def __str__(self):
        return self.method_title


