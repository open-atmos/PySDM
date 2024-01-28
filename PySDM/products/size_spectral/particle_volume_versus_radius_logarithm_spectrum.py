"""
n_V^e(ln(r)) particle volume spectrum per volume of air (uses natural logarithm),
i.e. volume of particles per volume of air having in the size range ln(r) to
ln(r) + dln(r)
"""

import numpy as np

from PySDM.products.impl.spectrum_moment_product import SpectrumMomentProduct


class ParticleVolumeVersusRadiusLogarithmSpectrum(SpectrumMomentProduct):
    def __init__(self, radius_bins_edges, name=None, unit="dimensionless", dry=False):
        super().__init__(name=name, unit=unit, attr_unit="m^3")
        self.radius_bins_edges = radius_bins_edges
        self.moment_0 = None
        self.moments = None
        self.attr = ("dry " if dry else "") + "volume"

    def register(self, builder):
        builder.request_attribute("volume")

        volume_bins_edges = builder.particulator.formulae.trivia.volume(
            self.radius_bins_edges
        )
        self.attr_bins_edges = builder.particulator.backend.Storage.from_ndarray(
            volume_bins_edges
        )

        super().register(builder)

        self.shape = (*builder.particulator.mesh.grid, len(self.attr_bins_edges) - 1)

    def _impl(self, **kwargs):
        vals = np.empty([self.particulator.mesh.n_cell, len(self.attr_bins_edges) - 1])
        self._recalculate_spectrum_moment(attr=self.attr, rank=1, filter_attr=self.attr)

        for i in range(vals.shape[1]):
            self._download_spectrum_moment_to_buffer(rank=1, bin_number=i)
            vals[:, i] = self.buffer.ravel()
            self._download_spectrum_moment_to_buffer(rank=0, bin_number=i)
            vals[:, i] *= self.buffer.ravel()

        vals *= 1 / np.diff(np.log(self.radius_bins_edges)) / self.particulator.mesh.dv
        return vals
