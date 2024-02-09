"""
mole fractions of aqueous-chemistry relevant compounds
"""

from PySDM.products.impl.moment_product import MomentProduct

DUMMY_SPECIFIC_GRAVITY = 44


class AqueousMoleFraction(MomentProduct):
    def __init__(self, key, unit="dimensionless", name=None):
        super().__init__(unit=unit, name=name)
        self.aqueous_chemistry = None
        self.key = key

    def register(self, builder):
        super().register(builder)
        self.aqueous_chemistry = self.particulator.dynamics["AqueousChemistry"]

    def _impl(self, **kwargs):
        attr = "moles_" + self.key

        self._download_moment_to_buffer(attr=attr, rank=0)
        conc = self.buffer.copy()

        self._download_moment_to_buffer(attr=attr, rank=1)
        tmp = self.buffer.copy()
        tmp[:] *= conc
        tmp[:] *= DUMMY_SPECIFIC_GRAVITY * self.formulae.constants.Md

        self._download_to_buffer(self.particulator.environment["rhod"])
        tmp[:] /= self.particulator.mesh.dv
        tmp[:] /= self.buffer
        tmp[:] = self.formulae.trivia.mixing_ratio_2_mole_fraction(
            tmp[:], specific_gravity=DUMMY_SPECIFIC_GRAVITY
        )
        return tmp
