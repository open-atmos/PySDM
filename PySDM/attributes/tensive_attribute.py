"""
Created at 11.05.2020
"""

from .base_attribute import BaseAttribute


class TensiveAttribute(BaseAttribute):

    def __init__(self, builder, name, extensive, dtype=float):
        super().__init__(builder, name, dtype)
        self.extensive = extensive
