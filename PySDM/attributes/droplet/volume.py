"""
Created at 11.05.2020
"""

from PySDM.attributes.tensive_attribute import TensiveAttribute


class Volume(TensiveAttribute):

    def __init__(self, particles_builder):
        super().__init__(particles_builder, name='volume', extensive=True)
