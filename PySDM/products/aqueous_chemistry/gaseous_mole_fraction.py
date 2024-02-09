"""
mole fractions of gaseous compounds relevant for aqueous chemistry
"""

from PySDM.dynamics.impl.chemistry_utils import GASEOUS_COMPOUNDS
from PySDM.products.impl.product import Product


class GaseousMoleFraction(Product):
    def __init__(self, key, unit="dimensionless", name=None):
        super().__init__(name=name, unit=unit)
        self.aqueous_chemistry = None
        self.compound = GASEOUS_COMPOUNDS[key]

    def register(self, builder):
        super().register(builder)
        self.aqueous_chemistry = self.particulator.dynamics["AqueousChemistry"]

    def _impl(self, **kwargs):
        tmp = self.formulae.trivia.mixing_ratio_2_mole_fraction(
            self.aqueous_chemistry.environment_mixing_ratios[self.compound],
            specific_gravity=self.aqueous_chemistry.specific_gravities[self.compound],
        )
        return tmp
