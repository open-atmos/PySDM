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
        self.grid = None

    def __del__(self):
        if hasattr(self, 'temp_dir'):
            self.temp_dir.cleanup()

    def init(self, settings):
        self.grid = settings.grid

    def _filepath(self, name: str, step: int = None):
        if step is None:
            filename = f"{name}.npy"
        else:
            filename = f"{name}_{step:06}.npy"
        path = os.path.join(self.dir_path, filename)
        return path

    def save(self, data: (float, np.ndarray), step: int, name: str):
        if isinstance(data, (int, float)):
            path = self._filepath(name)
            np.save(path, np.concatenate((() if step == 0 else np.load(path), (self.dtype(data),))))
        elif data.shape[0:2] == self.grid:
            np.save(self._filepath(name, step), data.astype(self.dtype))
        else:
            raise NotImplementedError()

    def load(self, name: str, step: int = None) -> np.ndarray:
        try:
            data = np.load(self._filepath(name, step))
        except FileNotFoundError:
            raise Storage.Exception()
        return data

