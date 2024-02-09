"""
particle temperature (test-use only for now, exemplifying intensive/extensive attribute logic)
"""

from PySDM.attributes.impl.intensive_attribute import IntensiveAttribute


class Temperature(IntensiveAttribute):
    def __init__(self, builder):
        super().__init__(builder, base="heat", name="temperature")
