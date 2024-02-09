"""
Provides radius bin-resolved average terminal velocity (average is particle-number weighted)
"""

import numpy as np

from PySDM.products.impl.spectrum_moment_product import SpectrumMomentProduct

ATTR = "terminal velocity"
RANK = 1


class RadiusBinnedNumberAveragedTerminalVelocity(SpectrumMomentProduct):
    def __init__(self, radius_bin_edges, name=None, unit="m/s"):
        super().__init__(name=name, unit=unit, attr_unit="m")
        self.radius_bin_edges = radius_bin_edges

    def register(self, builder):
        builder.request_attribute(ATTR)

        volume_bin_edges = builder.particulator.formulae.trivia.volume(
            self.radius_bin_edges
        )
        self.attr_bins_edges = builder.particulator.backend.Storage.from_ndarray(
            volume_bin_edges
        )

        super().register(builder)

        self.shape = (*builder.particulator.mesh.grid, len(self.attr_bins_edges) - 1)

    def _impl(self, **kwargs):
        vals = np.empty([self.particulator.mesh.n_cell, len(self.attr_bins_edges) - 1])

        self._recalculate_spectrum_moment(
            attr=ATTR,
            rank=RANK,
        )

        for i in range(vals.shape[1]):
            self._download_spectrum_moment_to_buffer(rank=RANK, bin_number=i)
            vals[:, i] = self.buffer.ravel()

        return np.squeeze(vals.reshape(self.shape))
