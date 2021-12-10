from PySDM.attributes.impl.derived_attribute import DerivedAttribute


class HydrogenIonConcentration(DerivedAttribute):
    def __init__(self, builder):
        self.pH = builder.get_attribute('pH')
        dependencies = [self.pH]
        super().__init__(builder, name='conc_H', dependencies=dependencies)

    def recalculate(self):
        self.data[:] = self.formulae.trivia.pH2H(self.pH.get().data)
