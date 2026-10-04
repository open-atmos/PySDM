"""logic around `PySDM.attributes.impl.dummy_attribute.DummyAttribute` - parent class
for do-nothing attributes"""

import numpy as np

from .attribute import Attribute


class DummyAttribute(Attribute):
    def __init__(self, particulator, name):
        super().__init__(particulator, name)

    def allocate(self, idx):
        super().allocate(idx)
        self.data.fill(np.nan)

    def get(self):
        return self.data
