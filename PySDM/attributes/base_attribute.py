"""
Created at 11.05.2020
"""

from .attribute import Attribute


class BaseAttribute(Attribute):

    def __init__(self, particles_builder, name, dtype=float, size=1):
        super().__init__(particles_builder, name=name, dtype=dtype, size=size)

    def init(self, data=None):
        if self.data is None:
            self.allocate()
        self.data.upload(data)
        self.mark_updated()
