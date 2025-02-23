# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np


class DummyStorage:
    def __init__(self):
        self.profiles = []

    def init(*_):  # pylint: disable=no-method-argument,no-self-argument
        pass

    def save(
        self, data: np.ndarray, step: int, name: str
    ):  # pylint: disable=unused-argument
        if name == "water_vapour_mixing_ratio_env":
            self.profiles.append(
                {"water_vapour_mixing_ratio_env": np.mean(data, axis=0)}
            )
