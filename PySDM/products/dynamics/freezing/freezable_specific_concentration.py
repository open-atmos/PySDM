import numpy as np
from PySDM.physics import constants as const
from PySDM.products.product import SpectrumMomentProduct


class FreezableSpecificConcentration(SpectrumMomentProduct):

    def __init__(self, temperature_bins_edges):
        super().__init__(
            name='Freezable specific concentration',
            unit=f"mg-1 K-1",
            description='Freezable specific concentration'
        )
        self.attr_bins_edges = temperature_bins_edges

    def register(self, builder):
        builder.request_attribute('freezing temperature')
        self.attr_bins_edges = builder.particulator.bck.Storage.from_ndarray(self.attr_bins_edges)
        super().register(builder)
        self.shape = (*builder.particulator.mesh.grid, len(self.attr_bins_edges) - 1)

    def get(self):
        vals = np.empty([self.particulator.mesh.n_cell, len(self.attr_bins_edges) - 1])
        self.recalculate_spectrum_moment(attr='volume', filter_attr='freezing temperature', rank=0)

        for i in range(vals.shape[1]):
            self.download_spectrum_moment_to_buffer(rank=0, bin_number=i)
            vals[:, i] = self.buffer.ravel()

        self.download_to_buffer(self.particulator.environment['rhod'])
        rhod = self.buffer.ravel()
        for i in range(len(self.attr_bins_edges) - 1):
            dT = abs(self.attr_bins_edges[i + 1] - self.attr_bins_edges[i])
            vals[:, i] /= rhod * dT * self.particulator.mesh.dv

        const.convert_to(vals, const.si.milligram**-1 / const.si.K)

        return np.squeeze(vals.reshape(self.shape))
