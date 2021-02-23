"""
Created at 14.05.2020
"""

from PySDM.attributes.intensive_attribute import IntensiveAttribute


class Temperature(IntensiveAttribute):

    def __init__(self, builder):
        super().__init__(builder, base='heat', name='temperature')
