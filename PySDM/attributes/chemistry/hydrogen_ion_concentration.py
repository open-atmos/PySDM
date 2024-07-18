"""
hydrogen ion concentration derived from pH
"""

import PySDM
from PySDM.attributes.impl.derived_attribute import DerivedAttribute


@PySDM.register_attribute(name="conc_H")
class HydrogenIonConcentration(DerivedAttribute):
    def __init__(self, builder):
        self.acidity = builder.get_attribute("pH")
        super().__init__(builder, name="conc_H", dependencies=(self.acidity,))

    def recalculate(self):
        self.data[:] = self.formulae.trivia.pH2H(self.acidity.get().data)
