"""
grid cell id attribute
"""

from PySDM.attributes.impl import CellAttribute, register_attribute


@register_attribute()
class CellId(CellAttribute):
    def __init__(self, builder):
        super().__init__(builder, name="cell id", dtype=int)

    def recalculate(self):
        pass
