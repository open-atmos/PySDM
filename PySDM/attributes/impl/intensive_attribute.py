"""
logic around `PySDM.attributes.impl.intensive_attribute.IntensiveAttribute` - parent class
 for all intensive attributes
"""

from .derived_attribute import DerivedAttribute


class IntensiveAttribute(DerivedAttribute):
    def __init__(self, particulator, name: str, base: str):
        self.volume = particulator.get_attribute("volume")
        self.base = particulator.get_attribute(base)
        super().__init__(particulator, name, dependencies=(self.volume, self.base))

    def recalculate(self):
        self.data.ratio(self.base.get(), self.volume.get())
