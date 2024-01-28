"""
grid-cell origin (multi-dimensional)
"""

from PySDM.attributes.impl.cell_attribute import CellAttribute


class CellOrigin(CellAttribute):
    def __init__(self, builder):
        super().__init__(
            builder,
            name="cell origin",
            dtype=int,
            n_vector_components=builder.particulator.mesh.dim,
        )
