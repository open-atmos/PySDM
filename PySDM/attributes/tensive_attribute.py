"""
Created at 11.05.2020
"""

from .base_attribute import BaseAttribute


class TensiveAttribute(BaseAttribute):

    def __init__(self, particles_builder, name, extensive, dtype=float):
        super().__init__(particles_builder, name, dtype)
        self.extensive = extensive
