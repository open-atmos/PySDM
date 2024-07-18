"""
position-within-cell attribute (multi-dimensional, values normalised to one)
"""

import PySDM
from PySDM.attributes.impl.cell_attribute import CellAttribute


@PySDM.register_attribute()
class PositionInCell(CellAttribute):
    def __init__(self, builder):
        super().__init__(
            builder,
            name="position in cell",
            n_vector_components=builder.particulator.mesh.dim,
        )
