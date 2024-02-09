"""
concentration of particles within a grid cell (either per-volume of per-mass-of-dry air,
 optionally restricted to a given size range)
"""

import numpy as np

from PySDM.products.impl.concentration_product import ConcentrationProduct


class ParticleConcentration(ConcentrationProduct):
    # pylint: disable=too-many-arguments
    def __init__(
        self,
        radius_range=(0, np.inf),
        specific=False,
        stp=False,
        name=None,
        unit="m^-3",
    ):
        self.radius_range = radius_range
        super().__init__(name=name, unit=unit, specific=specific, stp=stp)

    def _impl(self, **kwargs):
        self._download_moment_to_buffer(
            attr="water mass",
            rank=0,
            filter_range=(
                self.formulae.particle_shape_and_density.volume_to_mass(
                    self.formulae.trivia.volume(radius=self.radius_range[0])
                ),
                self.formulae.particle_shape_and_density.volume_to_mass(
                    self.formulae.trivia.volume(self.radius_range[1])
                ),
            ),
        )
        return super()._impl(**kwargs)


class ParticleSpecificConcentration(ParticleConcentration):
    def __init__(self, radius_range=(0, np.inf), name=None, unit="kg^-1"):
        super().__init__(radius_range=radius_range, specific=True, name=name, unit=unit)
