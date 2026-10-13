"""
hydrogen ion concentration derived from pH
"""

from PySDM.attributes.impl import DerivedAttribute, register_attribute


@register_attribute(name="conc_H")
class HydrogenIonConcentration(DerivedAttribute):
    def __init__(self, particulator):
        self.acidity = particulator.get_attribute("pH")
        super().__init__(particulator, name="conc_H", dependencies=(self.acidity,))

    def recalculate(self):
        self.data[:] = self.formulae.trivia.pH2H(self.acidity.get().data)
