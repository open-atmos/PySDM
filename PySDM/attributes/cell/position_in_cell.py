"""
Created at 11.05.2020
"""

from PySDM.attributes.base_attribute import BaseAttribute


class PositionInCell(BaseAttribute):

    def __init__(self, particles_builder):
        super().__init__(particles_builder, name='position in cell', size=particles_builder.particles.mesh.dim)
