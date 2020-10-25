"""
Created at 02.10.2019
"""

import os
import tempfile
import numpy as np
from pathlib import Path


class Storage:
    class Exception(BaseException):
        pass

    def __init__(self, dtype=np.float32, path=None):
        if path is None:
            self.temp_dir = tempfile.TemporaryDirectory()
            self.dir_path = self.temp_dir.name
        else:
            Path(path).mkdir(parents=True, exist_ok=True)
            self.dir_path = Path(path).absolute()
        self.dtype = dtype

    def __del__(self):
        if hasattr(self, 'temp_dir'):
            self.temp_dir.cleanup()

    def init(self, settings):
        self.grid = settings.grid

    def _filepath(self, step: int, name: str):
        path = os.path.join(self.dir_path, f"{step:06}_{name}.npy")
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

