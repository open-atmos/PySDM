"""
super-particle multiplicity (aka weighting factor) - the number of real-world particles
 represented in the simulation with a given super particle
"""
from PySDM.attributes.impl.base_attribute import BaseAttribute


class Multiplicities(BaseAttribute):
    def __init__(self, builder):
        super().__init__(builder, name="n", dtype=int)
