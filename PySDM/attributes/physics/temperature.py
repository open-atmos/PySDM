"""
particle temperature (test-use only for now, exemplifying intensive/extensive attribute logic)
"""

from PySDM.attributes.impl import IntensiveAttribute, register_attribute


@register_attribute()
class Temperature(IntensiveAttribute):
    def __init__(self, particulator):
        super().__init__(particulator, base="heat", name="temperature")
