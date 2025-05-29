"""
position-within-cell attribute (multi-dimensional, values normalised to one)
"""

from PySDM.attributes.impl import CellAttribute, register_attribute


@register_attribute()
class PositionInCell(CellAttribute):
    def __init__(self, builder):
        super().__init__(
            builder,
            name="position in cell",
            n_vector_components=builder.particulator.mesh.dim,
        )
