"""
Created at 11.05.2020
"""

from PySDM.attributes.cell_attribute import CellAttribute


class PositionInCell(CellAttribute):

    def __init__(self, builder):
        super().__init__(builder, name='position in cell', size=builder.core.mesh.dim)
