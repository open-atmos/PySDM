from .attribute import Attribute
import numpy as np


class DummyAttribute(Attribute):
    def __init__(self, builder):
        super().__init__(builder, '')

    def allocate(self, idx):
        super().allocate(idx)
        self.data[:] = np.nan

    def get(self):
        return self.data
