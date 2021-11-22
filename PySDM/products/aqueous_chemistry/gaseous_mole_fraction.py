from PySDM.products.impl.product import Product
from PySDM.physics.aqueous_chemistry.support import GASEOUS_COMPOUNDS, SPECIFIC_GRAVITY


class GaseousMoleFraction(Product):
    def __init__(self, key, unit='dimensionless', name=None):
        super().__init__(name=name, unit=unit)
        self.aqueous_chemistry = None
        self.compound = GASEOUS_COMPOUNDS[key]

    def register(self, builder):
        super().register(builder)
        self.aqueous_chemistry = self.particulator.dynamics['AqueousChemistry']

    def _impl(self, **kwargs):
        tmp = self.formulae.trivia.mixing_ratio_2_mole_fraction(
            self.aqueous_chemistry.environment_mixing_ratios[self.compound],
            specific_gravity=SPECIFIC_GRAVITY[self.compound]
        )
        return tmp
