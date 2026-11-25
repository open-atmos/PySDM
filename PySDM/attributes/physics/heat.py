"""
particle heat content (test-use only for now, exemplifying intensive/extensive attribute logic)
"""

from PySDM.attributes.impl import ExtensiveAttribute, register_attribute


@register_attribute()
class Heat(ExtensiveAttribute):
    def __init__(self, particulator):
        super().__init__(particulator, name="heat")
