"""
Created at 09.07.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import time


class Stats:
    def __init__(self):
        self.cpu_times = []
        self.wall_times = []
        self.cpu_t0 = None
        self.wall_t0 = None

    def __enter__(self):
        self.cpu_t0 = time.process_time()
        self.wall_t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        cpu_t1 = time.process_time()
        wall_t1 = time.perf_counter()

        self.cpu_times.append(cpu_t1 - self.cpu_t0)
        self.wall_times.append(wall_t1 - self.wall_t0)
