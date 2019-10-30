import os
import numpy as np
import tempfile


class Storage:
    class Exception(BaseException):
        pass

    def __init__(self):
        self.reinit()

    def reinit(self, _=None):
        self.tempdir = tempfile.TemporaryDirectory()

    def _filepath(self, step: int):
        path = os.path.join(self.tempdir.name, f"{step}.npy")
        return path

    def save(self, state: np.ndarray, step: int):
        np.save(self._filepath(step), state)

    def load(self, step):
        try :
            return np.load(self._filepath(step))
        except FileNotFoundError:
            raise Storage.Exception()

#    def create_netcdf():
#        raise NotInmplementedError()