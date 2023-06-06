import os
import tempfile
from pathlib import Path

import numpy as np


class Storage:
    class Exception(BaseException):
        pass

    def __init__(self, dtype=np.float32, path=None):
        self.temp_dir = None
        if path is None:
            self.setup_temporary_directory()
        else:
            Path(path).mkdir(parents=True, exist_ok=True)
            self.dir_path = Path(path).absolute()
        self.dtype = dtype
        self.grid = None
        self._data_range = None

    def __del__(self):
        self.cleanup()

    def cleanup(self):
        if self.temp_dir is not None:
            self.temp_dir.cleanup()

    def setup_temporary_directory(self):
        self.cleanup()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = self.temp_dir.name

    def init(self, settings):
        self.grid = settings.grid
        self._data_range = {}
        if self.temp_dir is not None and any(os.scandir(self.temp_dir.name)):
            self.setup_temporary_directory()

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
            np.save(
                path,
                np.concatenate(
                    (() if step == 0 else np.load(path), (self.dtype(data),))
                ),
            )
        elif data.shape[0:2] == self.grid:
            np.save(self._filepath(name, step), data.astype(self.dtype))
        else:
            raise NotImplementedError()

        if name not in self._data_range:
            self._data_range[name] = (np.inf, -np.inf)
        just_nans = np.isnan(data).all()
        self._data_range[name] = (
            min(
                self._data_range[name][0] if just_nans else np.nanmin(data),
                self._data_range[name][0],
            ),
            max(
                self._data_range[name][1] if just_nans else np.nanmax(data),
                self._data_range[name][1],
            ),
        )

    def data_range(self, name):
        return self._data_range[name]

    def load(self, name: str, step: int = None) -> np.ndarray:
        try:
            data = np.load(self._filepath(name, step))
        except FileNotFoundError as err:
            raise Storage.Exception() from err
        return data
