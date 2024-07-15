"""
grid-cell origin (multi-dimensional)
"""

import PySDM
from PySDM.attributes.impl.cell_attribute import CellAttribute


@PySDM.attribute()
class CellOrigin(CellAttribute):
    def __init__(self, builder):
        super().__init__(
            builder,
            name="cell origin",
            dtype=int,
            n_vector_components=builder.particulator.mesh.dim,
        )
