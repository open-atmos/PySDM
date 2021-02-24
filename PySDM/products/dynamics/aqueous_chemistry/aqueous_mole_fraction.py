from ...product import MomentProduct
from ....physics.formulae import mixing_ratio_2_mole_fraction
from ....dynamics.aqueous_chemistry.aqueous_chemistry import SPECIFIC_GRAVITY
from ....physics.constants import convert_to, ppb, Md


class AqueousMoleFraction(MomentProduct):
    def __init__(self, compound):
        super().__init__(
            name=f'aq_{compound}_ppb',
            unit='ppb',
            description=f'aqueous {compound} mole fraction',
            scale=None,
            range=None
        )
        self.aqueous_chemistry = None
        self.compound = compound

    def register(self, builder):
        super().register(builder)
        self.aqueous_chemistry = self.core.dynamics['AqueousChemistry']

    def get(self):
        attr = 'moles_' + self.compound

        self.download_moment_to_buffer(attr, rank=0)
        conc = self.buffer.copy()

        self.download_moment_to_buffer(attr, rank=1)
        tmp = self.buffer.copy()
        tmp[:] *= conc
        tmp[:] *= SPECIFIC_GRAVITY[self.compound] * Md

        self.download_to_buffer(self.core.environment['rhod'])
        tmp[:] /= self.core.mesh.dv
        tmp[:] /= self.buffer
        tmp[:] = mixing_ratio_2_mole_fraction(tmp[:], specific_gravity=SPECIFIC_GRAVITY[self.compound])
        convert_to(tmp, ppb)
        return tmp
