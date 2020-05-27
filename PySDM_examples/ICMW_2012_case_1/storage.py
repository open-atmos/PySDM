"""
Created at 02.10.2019

@author: Sylwester Arabas
"""

import os, tempfile, shutil
import numpy as np


class Storage:
    class Exception(BaseException):
        pass

    def __init__(self, dtype=np.float32):
        self.createdir()
        self.dtype = dtype

    def init(self, setup):
        self.createdir()
        self.grid = setup.grid

    def createdir(self):
        self.tempdir = tempfile.TemporaryDirectory()

    def _filepath(self, step: int, name: str):
        path = os.path.join(self.tempdir.name, f"{step:06}_{name}.npy")
        return path

    def save(self, data: np.ndarray, step: int, name: str):
        assert (data.shape[0:2] == self.grid)
        np.save(self._filepath(step, name), data.astype(self.dtype))

    def load(self, step: int, name: str) -> np.ndarray:
        try:
            data = np.load(self._filepath(step, name))
        except FileNotFoundError:
            raise Storage.Exception()
        return data

