"""
Created at 02.10.2019

@author: Sylwester Arabas
"""

import os
import numpy as np
import tempfile


class Storage:
    class Exception(BaseException):
        pass

    def __init__(self, dtype=np.float32):
        self.createdir()
        self.dtype = dtype

    def init(self, setup):
        self.tempdir.cleanup()
        self.createdir()
        # TODO: dump serialised setup?

    def createdir(self):
        self.tempdir = tempfile.TemporaryDirectory()

    def _filepath(self, step: int, name: str):
        path = os.path.join(self.tempdir.name, f"{step:06}_{name}.npy")
        return path

    def save(self, data: np.ndarray, step: int, name: str):
        np.save(self._filepath(step, name), data.astype(self.dtype))

    def load(self, step: int, name: str) -> np.ndarray:
        try:
            data = np.load(self._filepath(step, name))
        except FileNotFoundError:
            raise Storage.Exception()
        return data

#    def create_netcdf():
#        raise NotInmplementedError()

