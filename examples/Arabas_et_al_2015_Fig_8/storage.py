import os
import numpy as np
import tempfile


class Storage:
    class Exception(BaseException):
        pass

    def __init__(self):
        self.createdir()

    def init(self, setup):
        self.tempdir.cleanup()
        self.createdir()
        # TODO: dump serialised setup?

    def createdir(self):
        self.tempdir = tempfile.TemporaryDirectory()

    def _filepath(self, step: int):
        path = os.path.join(self.tempdir.name, f"{step:06}.npy")
        return path

    def save(self, data: np.ndarray, step: int):
        np.save(self._filepath(step), data)

    def load(self, step: int) -> np.ndarray:
        try:
            data = np.load(self._filepath(step))
        except FileNotFoundError:
            raise Storage.Exception()
        return data

#    def create_netcdf():
#        raise NotInmplementedError()
