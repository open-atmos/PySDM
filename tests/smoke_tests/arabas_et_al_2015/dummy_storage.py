# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np


class DummyStorage:
    def __init__(self):
        self.profiles = []

    def init(*_):  # pylint: disable=no-method-argument
        pass

    def save(self, data: np.ndarray, step: int, name: str):  # pylint: disable=unused-argument
        if name == "qv_env":
            self.profiles.append({"qv_env": np.mean(data, axis=0)})
