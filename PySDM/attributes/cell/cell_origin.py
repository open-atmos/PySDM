"""
Created at 11.05.2020
"""

from PySDM.attributes.base_attribute import BaseAttribute


class CellOrigin(BaseAttribute):

    def __init__(self, builder):
        super().__init__(builder, name='cell origin', dtype=int, size=builder.core.mesh.dim)
