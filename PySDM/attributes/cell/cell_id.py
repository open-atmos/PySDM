"""
Created at 11.05.2020
"""

from PySDM.attributes.base_attribute import BaseAttribute


class CellID(BaseAttribute):

    def __init__(self, particles_builder):
        super().__init__(particles_builder, name='cell id', dtype=int)
