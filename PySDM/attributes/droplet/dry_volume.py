"""
Created at 11.05.2020

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from PySDM.attributes.tensive_attribute import TensiveAttribute


class DryVolume(TensiveAttribute):

    def __init__(self, particles_builder):
        super().__init__(particles_builder, name='dry volume', extensive=True)
