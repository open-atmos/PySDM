"""
grid cell id attribute
"""

import PySDM
from PySDM.attributes.impl.cell_attribute import CellAttribute


@PySDM.attribute()
class CellId(CellAttribute):
    def __init__(self, builder):
        super().__init__(builder, name="cell id", dtype=int)

    def recalculate(self):
        pass
