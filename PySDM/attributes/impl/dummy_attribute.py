""" logic around `PySDM.attributes.impl.dummy_attribute.DummyAttribute` - parent class
for do-nothing attributes """

import warnings

import numpy as np

from .attribute import Attribute


class DummyAttribute(Attribute):
    def __init__(self, builder, name):
        super().__init__(builder, name)

    def allocate(self, idx):
        super().allocate(idx)
        self.data[:] = np.nan

    def get(self):
        return self.data


def make_dummy_attribute_factory(name, warn=False):
    def _factory(builder):
        return DummyAttribute(builder, name=name)

    if warn:
        warnings.warn(
            f"dummy implementation used for requested attribute named '{name}'"
        )

    return _factory
