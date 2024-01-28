"""
common code for products computing **binned** statistical moments
 (e.g., dry radius spectrum in each grid cell)
"""

from abc import ABC

from PySDM.products.impl.product import Product


class SpectrumMomentProduct(ABC, Product):
    def __init__(self, name, unit, attr_unit):
        super().__init__(name=name, unit=unit)
        self.attr_bins_edges = None
        self.attr_unit = attr_unit
        self.moment_0 = None
        self.moments = None

    def register(self, builder):
        super().register(builder)
        self.moment_0 = self.particulator.Storage.empty(
            (len(self.attr_bins_edges) - 1, self.particulator.mesh.n_cell), dtype=float
        )
        self.moments = self.particulator.Storage.empty(
            (len(self.attr_bins_edges) - 1, self.particulator.mesh.n_cell), dtype=float
        )
        _ = self._parse_unit(self.attr_unit)

    def _recalculate_spectrum_moment(
        self,
        *,
        attr,
        rank,
        filter_attr="volume",
        weighting_attribute="volume",
        weighting_rank=0,
    ):
        self.particulator.spectrum_moments(
            moment_0=self.moment_0,
            moments=self.moments,
            attr=attr,
            rank=rank,
            attr_bins=self.attr_bins_edges,
            attr_name=filter_attr,
            weighting_attribute=weighting_attribute,
            weighting_rank=weighting_rank,
        )

    def _download_spectrum_moment_to_buffer(self, rank, bin_number):
        if rank == 0:  # TODO #217
            self._download_to_buffer(self.moment_0[bin_number, :])
        else:
            self._download_to_buffer(self.moments[bin_number, :])
