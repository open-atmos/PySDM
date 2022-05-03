"""
wet radius-binned particle size spectra (per mass of dry air or per volume of air)
"""
from abc import ABC

import numpy as np

from PySDM.products.impl.spectrum_moment_product import SpectrumMomentProduct


class ParticleSizeSpectrum(SpectrumMomentProduct, ABC):
    def __init__(
        self, *, radius_bins_edges, name, unit, dry=False, normalise_by_dv=False
    ):
        self.volume_attr = "dry volume" if dry else "volume"
        self.radius_bins_edges = radius_bins_edges
        self.normalise_by_dv = normalise_by_dv
        super().__init__(name=name, unit=unit, attr_unit="m")

    def register(self, builder):
        builder.request_attribute(self.volume_attr)

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
        self._recalculate_spectrum_moment(
            attr=self.volume_attr, rank=1, filter_attr=self.volume_attr
        )

        for i in range(vals.shape[1]):
            self._download_spectrum_moment_to_buffer(rank=0, bin_number=i)
            vals[:, i] = self.buffer.ravel()

        if self.normalise_by_dv:
            vals[:] /= self.particulator.mesh.dv

        self._download_to_buffer(self.particulator.environment["rhod"])
        rhod = self.buffer.ravel()
        for i in range(len(self.attr_bins_edges) - 1):
            dr = self.formulae.trivia.radius(
                volume=self.attr_bins_edges[i + 1]
            ) - self.formulae.trivia.radius(volume=self.attr_bins_edges[i])
            vals[:, i] /= rhod * dr

        return np.squeeze(vals.reshape(self.shape))


class ParticleSizeSpectrumPerMass(ParticleSizeSpectrum):
    def __init__(self, radius_bins_edges, dry=False, name=None, unit="kg^-1 m^-1"):
        super().__init__(
            radius_bins_edges=radius_bins_edges,
            dry=dry,
            normalise_by_dv=True,
            name=name,
            unit=unit,
        )


class ParticleSizeSpectrumPerVolume(ParticleSizeSpectrum):
    def __init__(self, radius_bins_edges, dry=False, name=None, unit="m^-3 m^-1"):
        super().__init__(
            radius_bins_edges=radius_bins_edges,
            dry=dry,
            normalise_by_dv=False,
            name=name,
            unit=unit,
        )
