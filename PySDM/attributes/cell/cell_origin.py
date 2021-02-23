"""
Created at 11.05.2020
"""

from PySDM.attributes.cell_attribute import CellAttribute


class CellOrigin(CellAttribute):

    def __init__(self, builder):
        super().__init__(builder, name='cell origin', dtype=int, size=builder.core.mesh.dim)
