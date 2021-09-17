from ...product import MomentProduct
from ....physics.constants import convert_to, ppb, Md


class AqueousMoleFraction(MomentProduct):
    def __init__(self, key):
        super().__init__(
            name=f'aq_{key}_ppb',
            unit='ppb',
            description=f'aqueous {key} mole fraction'
        )
        self.aqueous_chemistry = None
        self.key = key

    def register(self, builder):
        super().register(builder)
        self.aqueous_chemistry = self.particulator.dynamics['AqueousChemistry']

    def get(self):
        attr = 'moles_' + self.key

        self.download_moment_to_buffer(attr, rank=0)
        conc = self.buffer.copy()

        self.download_moment_to_buffer(attr, rank=1)
        tmp = self.buffer.copy()
        tmp[:] *= conc
        tmp[:] *= 44 * Md

        self.download_to_buffer(self.particulator.environment['rhod'])
        tmp[:] /= self.particulator.mesh.dv
        tmp[:] /= self.buffer
        tmp[:] = self.formulae.trivia.mixing_ratio_2_mole_fraction(tmp[:], specific_gravity=44)
        convert_to(tmp, ppb)
        return tmp
