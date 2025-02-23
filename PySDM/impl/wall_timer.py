"""
context manager automating wall-time counting using Python's basic
 [time.perf_counter()](https://docs.python.org/3/library/time.html#time.perf_counter)
"""

import time


class WallTimer:
    @staticmethod
    def __clock():
        return time.perf_counter()

    def __init__(self):
        self.time = None

    def __enter__(self):
        self.time = self.__clock()

    def __exit__(self, *_):
        self.time *= -1
        self.time += self.__clock()
