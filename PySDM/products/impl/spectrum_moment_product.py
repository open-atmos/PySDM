from abc import ABC
from PySDM.products.impl.product import Product


class SpectrumMomentProduct(ABC, Product):
    def __init__(self, name, unit):
        super().__init__(name=name, unit=unit)
        self.attr_bins_edges = None
        self.moment_0 = None
        self.moments = None

    def register(self, builder):
        super().register(builder)
        self.moment_0 = self.particulator.Storage.empty(
            (len(self.attr_bins_edges) - 1, self.particulator.mesh.n_cell), dtype=float)
        self.moments = self.particulator.Storage.empty(
            (len(self.attr_bins_edges) - 1, self.particulator.mesh.n_cell), dtype=float)

    def _recalculate_spectrum_moment(
        self, attr,
        rank, filter_attr='volume',
        weighting_attribute='volume', weighting_rank=0
    ):
        self.particulator.spectrum_moments(
            self.moment_0, self.moments, attr, rank, self.attr_bins_edges,
            attr_name=filter_attr,
            weighting_attribute=weighting_attribute, weighting_rank=weighting_rank
        )

    def _download_spectrum_moment_to_buffer(self, rank, bin_number):
        if rank == 0:  # TODO #217
            self._download_to_buffer(self.moment_0[bin_number, :])
        else:
            self._download_to_buffer(self.moments[bin_number, :])
