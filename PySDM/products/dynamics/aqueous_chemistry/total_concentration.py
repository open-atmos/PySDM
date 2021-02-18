from ...product import MomentProduct
from ....physics.formulae import mixing_ratio_2_mole_fraction
from ....dynamics.aqueous_chemistry.aqueous_chemistry import eps_SO2


class TotalConcentration(MomentProduct):
    def __init__(self, compound):
        super().__init__(
            name=f'{compound}_tot_conc',
            unit='ppb',
            description=f'total (gas + aqueous) {compound} mole fraction',
            scale=None,
            range=None
        )
        self.aqueous_chemistry = None
        self.compound = compound

    def register(self, builder):
        super().register(builder)
        self.aqueous_chemistry = self.core.dynamics['AqueousChemistry']

    def get(self):
        # TODO: add amount from droplets
        return mixing_ratio_2_mole_fraction(self.aqueous_chemistry.environment_mixing_ratios[self.compound],
                                            specific_gravity=eps_SO2)  # TODO !!!!
