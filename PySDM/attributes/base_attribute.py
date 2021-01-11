"""
Created at 11.05.2020
"""

from .attribute import Attribute


class BaseAttribute(Attribute):

    def __init__(self, builder, name, dtype=float, size=1):
        super().__init__(builder, name=name, dtype=dtype, size=size)

    def init(self, data):
        self.data.upload(data)
        self.mark_updated()
