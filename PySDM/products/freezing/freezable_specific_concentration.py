"""
freezing-temperature binned specific concentration of particles
"""

import numpy as np

from PySDM.products.impl.spectrum_moment_product import SpectrumMomentProduct


class FreezableSpecificConcentration(SpectrumMomentProduct):
    def __init__(self, temperature_bins_edges, name=None, unit="kg^-1 K^-1"):
        super().__init__(name=name, unit=unit, attr_unit="K")
        self.attr_bins_edges = temperature_bins_edges

    def register(self, builder):
        builder.request_attribute("freezing temperature")
        particulator = builder.particulator
        self.attr_bins_edges = particulator.backend.Storage.from_ndarray(
            self.attr_bins_edges
        )
        super().register(builder)
        self.shape = (*particulator.mesh.grid, len(self.attr_bins_edges) - 1)

    def _impl(self, **kwargs):
        vals = np.empty([self.particulator.mesh.n_cell, len(self.attr_bins_edges) - 1])
        self._recalculate_spectrum_moment(
            attr="volume", filter_attr="freezing temperature", rank=0
        )

        for i in range(vals.shape[1]):
            self._download_spectrum_moment_to_buffer(rank=0, bin_number=i)
            vals[:, i] = self.buffer.ravel()

        self._download_to_buffer(self.particulator.environment["rhod"])
        rhod = self.buffer.ravel()
        for i in range(len(self.attr_bins_edges) - 1):
            dT = abs(self.attr_bins_edges[i + 1] - self.attr_bins_edges[i])
            vals[:, i] /= rhod * dT * self.particulator.mesh.dv

        return np.squeeze(vals.reshape(self.shape))
