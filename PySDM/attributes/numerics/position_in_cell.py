"""
position-within-cell attribute (multi-dimensional, values normalised to one)
"""

from PySDM.attributes.impl import CellAttribute, register_attribute


@register_attribute()
class PositionInCell(CellAttribute):
    def __init__(self, particulator):
        super().__init__(
            particulator,
            name="position in cell",
            n_vector_components=particulator.mesh.dim,
        )
