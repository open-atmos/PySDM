import numpy as np
from PySDM.particulator import Particulator


class Observer:
    def notify(self)->None:
        raise NotImplementedError


class Progress(Observer):
    def __init__(self, max_steps):
        self.max_steps = max_steps
        self.steps = 0

    def notify(self) -> None:
        self.steps += 1
        progress = 100 * (self.steps / self.max_steps)
        print(f"Progress: {progress:.0f}%    ", end="\r")
        if self.steps == self.max_steps:
            print()


class Logger(Observer):
    def __init__(self, particulator: Particulator):
        self.particulator = particulator

    def notify(self) -> None:
        n = self.particulator.attributes["n"].to_ndarray()
        print(n[:10])
