"""
n(V) particle volume spectrum per volume of air,
i.e. number of particles per volume of air having in the size range bin
"""

import numpy as np

from PySDM.products.impl.spectrum_moment_product import SpectrumMomentProduct


class NumberSizeSpectrum(SpectrumMomentProduct):
    def __init__(self, radius_bins_edges, name=None, unit="m^-3"):
        super().__init__(name=name, unit=unit, attr_unit="m")
        self.radius_bins_edges = radius_bins_edges
        self.moment_0 = None
        self.moments = None
        self.attr = "volume"

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
            self._download_spectrum_moment_to_buffer(rank=0, bin_number=i)
            vals[:, i] = self.buffer.ravel()

        vals *= 1 / self.particulator.mesh.dv
        return vals
