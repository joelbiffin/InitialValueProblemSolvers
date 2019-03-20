import time


class TimedSolver(object):
    """ Contains some helpful functions for analysing runtime of algorithms"""

    time: list

    def start_solve_time(self):
        self.time[0] = time.time()


    def end_solve_time(self):
        self.time[1] = time.time()


    @property
    def solve_time(self):
        return self.time[1] - self.time[0]


