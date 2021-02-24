from ...product import Product
from ....physics.formulae import mixing_ratio_2_mole_fraction
from ....dynamics.aqueous_chemistry.aqueous_chemistry import SPECIFIC_GRAVITY
from ....physics.constants import convert_to, ppb


class GaseousMoleFraction(Product):
    def __init__(self, compound):
        super().__init__(
            name=f'gas_{compound}_ppb',
            unit='ppb',
            description=f'gaseous {compound} mole fraction',
            scale=None,
            range=None
        )
        self.aqueous_chemistry = None
        self.compound = compound

    def register(self, builder):
        super().register(builder)
        self.aqueous_chemistry = self.core.dynamics['AqueousChemistry']

    def get(self):
        tmp = mixing_ratio_2_mole_fraction(
            self.aqueous_chemistry.environment_mixing_ratios[self.compound],
            specific_gravity=SPECIFIC_GRAVITY[self.compound]
        )
        convert_to(tmp, ppb)
        return tmp

