"""
position-within-cell attribute (multi-dimensional, values normalised to one)
"""

from PySDM.attributes.impl.cell_attribute import CellAttribute


class PositionInCell(CellAttribute):
    def __init__(self, builder):
        super().__init__(
            builder,
            name="position in cell",
            n_vector_components=builder.particulator.mesh.dim,
        )
