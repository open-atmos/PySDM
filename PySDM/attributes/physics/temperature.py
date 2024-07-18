"""
particle temperature (test-use only for now, exemplifying intensive/extensive attribute logic)
"""

import PySDM
from PySDM.attributes.impl.intensive_attribute import IntensiveAttribute


@PySDM.register_attribute()
class Temperature(IntensiveAttribute):
    def __init__(self, builder):
        super().__init__(builder, base="heat", name="temperature")
