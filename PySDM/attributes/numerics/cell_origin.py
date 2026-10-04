"""
grid-cell origin (multi-dimensional)
"""

from PySDM.attributes.impl import CellAttribute, register_attribute


@register_attribute()
class CellOrigin(CellAttribute):
    def __init__(self, particulator):
        super().__init__(
            particulator,
            name="cell origin",
            dtype=int,
            n_vector_components=particulator.mesh.dim,
        )
