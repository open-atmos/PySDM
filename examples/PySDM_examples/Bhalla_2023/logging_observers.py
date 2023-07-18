from logging import warning

import numpy as np

from PySDM.particulator import Particulator


class Observer:
    def notify(self) -> None:
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
        raise NotImplementedError()


class WarnVelocityDiff(Logger):
    def __init__(self, particulator: Particulator, threshold=1e-10):
        super().__init__(particulator)
        self.threshold = threshold

    def notify(self) -> None:
        fall_vel = self.particulator.attributes["fall velocity"].to_ndarray()

        terminal_vel = self.particulator.attributes["terminal velocity"].to_ndarray()

        rms_error = np.sqrt(np.mean((fall_vel - terminal_vel) ** 2))

        if rms_error > self.threshold:
            warning(
                f"The difference in fall velocity and terminal velocity is too high ({rms_error:.1e} > {self.threshold:.1e})"
            )
