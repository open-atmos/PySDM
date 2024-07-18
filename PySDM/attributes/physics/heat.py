"""
particle heat content (test-use only for now, exemplifying intensive/extensive attribute logic)
"""

import PySDM
from PySDM.attributes.impl.extensive_attribute import ExtensiveAttribute


@PySDM.register_attribute()
class Heat(ExtensiveAttribute):
    def __init__(self, builder):
        super().__init__(builder, name="heat")
