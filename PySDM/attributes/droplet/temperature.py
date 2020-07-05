"""
Created at 14.05.2020
"""

from PySDM.attributes.tensive_attribute import TensiveAttribute


class Temperature(TensiveAttribute):

    def __init__(self, particles_builder):
        super().__init__(particles_builder, name='temperature', extensive=False)
