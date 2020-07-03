"""
Created at 30.04.2020
"""


class Incrementation:
    def __init__(self, start=44, step=1):
        self.seed = start
        self.step = step

    def __call__(self):
        self.seed += self.step
        return self.seed
