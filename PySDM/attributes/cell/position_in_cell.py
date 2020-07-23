"""
Created at 11.05.2020
"""

from PySDM.attributes.base_attribute import BaseAttribute


class PositionInCell(BaseAttribute):

    def __init__(self, builder):
        super().__init__(builder, name='position in cell', size=builder.core.mesh.dim)
