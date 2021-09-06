from .attribute import Attribute
import numpy as np


class DummyAttribute(Attribute):
    def __init__(self, builder, name):
        super().__init__(builder, name)

    def allocate(self, idx):
        super().allocate(idx)
        self.data[:] = np.nan

    def get(self):
        return self.data


def DummyAttributeImpl(name):
    def _constructor(pb):
        return DummyAttribute(pb, name=name)
    return _constructor
