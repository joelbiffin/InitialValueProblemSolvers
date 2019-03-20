import csv
import numpy as np


class Solution(object):

    time_mesh: np.array
    value_mesh: np.array
    dimension: int


    def __init__(self, time_mesh, value_mesh, method_title="Unknown"):
        self.time_mesh = np.trim_zeros(time_mesh, "b")
        self.value_mesh = value_mesh[:self.time_mesh.size]
        self.dimension = self.value_mesh.shape[1]
        self.method_title = method_title

    def __str__(self):
        return self.method_title


    def write_to_csv(self, filename, components):
        output_columns = [self.time_mesh]

        for component in components:
            output_columns.append(self.value_mesh[:, component])

        output_np = np.array(output_columns)

        with open(filename, "w") as output_file:
            writer = csv.writer(output_file)

            for i in range(0, self.time_mesh.size):
                writer.writerow(output_np[j][i] for j in range(0, len(components) + 1))

        return






