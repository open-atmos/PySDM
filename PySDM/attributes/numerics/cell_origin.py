"""
grid-cell origin (multi-dimensional)
"""

from PySDM.attributes.impl import CellAttribute, register_attribute


@register_attribute()
class CellOrigin(CellAttribute):
    def __init__(self, builder):
        super().__init__(
            builder,
            name="cell origin",
            dtype=int,
            n_vector_components=builder.particulator.mesh.dim,
        )
