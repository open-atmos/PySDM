"""
super-particle multiplicity (aka weighting factor) - the number of real-world particles
 represented in the simulation with a given super particle
"""

import numpy as np
import PySDM
from PySDM.attributes.impl.base_attribute import BaseAttribute


@PySDM.register_attribute()
class Multiplicity(BaseAttribute):
    TYPE = np.int64
    MAX_VALUE = np.iinfo(TYPE).max

    def __init__(self, builder):
        super().__init__(builder, name="multiplicity", dtype=Multiplicity.TYPE)
