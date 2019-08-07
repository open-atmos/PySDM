"""
Created at 09.07.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import time


class Stats:
    def __init__(self):
        self.times = []
        self.t0 = 0.

    def __enter__(self):
        self.t0 = time.process_time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        t1 = time.process_time()
        self.times.append(t1 - self.t0)
