"""
wet- or dry-radius binned particle size spectra (per mass of dry air or per volume of air)
"""

from abc import ABC

import numpy as np

from PySDM.products.impl.spectrum_moment_product import SpectrumMomentProduct
from PySDM.products.impl.concentration_product import ConcentrationProduct


class ParticleSizeSpectrum(SpectrumMomentProduct, ABC):
    def __init__(
        self,
        *,
        stp: bool,
        radius_bins_edges,
        name: str,
        unit: str,
        dry: bool = False,
        specific: bool = False,
    ):
        ConcentrationProduct.check_ctor_arguments(specific, stp)
        self.volume_attr = "dry volume" if dry else "volume"
        self.radius_bins_edges = radius_bins_edges
        self.specific = specific
        self.stp = stp
        self.rho_stp = None
        super().__init__(name=name, unit=unit, attr_unit="m^3")

    def register(self, builder):
        builder.request_attribute(self.volume_attr)

        volume_bins_edges = builder.particulator.formulae.trivia.volume(
            np.asarray(self.radius_bins_edges)
        )
        self.attr_bins_edges = builder.particulator.backend.Storage.from_ndarray(
            volume_bins_edges
        )

        super().register(builder)

        self.shape = (*builder.particulator.mesh.grid, len(self.attr_bins_edges) - 1)
        self.rho_stp = builder.formulae.constants.rho_STP

    def _impl(self, **kwargs):
        vals = np.empty([self.particulator.mesh.n_cell, len(self.attr_bins_edges) - 1])
        self._recalculate_spectrum_moment(
            attr=self.volume_attr, rank=1, filter_attr=self.volume_attr
        )

        for i in range(vals.shape[1]):
            self._download_spectrum_moment_to_buffer(rank=0, bin_number=i)
            vals[:, i] = self.buffer.ravel()

        vals[:] /= self.particulator.mesh.dv

        if self.specific or self.stp:
            self._download_to_buffer(self.particulator.environment["rhod"])

        for i in range(len(self.attr_bins_edges) - 1):
            dr = self.formulae.trivia.radius(
                volume=self.attr_bins_edges[i + 1]
            ) - self.formulae.trivia.radius(volume=self.attr_bins_edges[i])
            vals[:, i] /= dr
            if self.specific or self.stp:
                vals[:, i] /= self.buffer.ravel()
                if self.stp:
                    vals[:, i] *= self.rho_stp

        return np.squeeze(vals.reshape(self.shape))


class ParticleSizeSpectrumPerMassOfDryAir(ParticleSizeSpectrum):
    def __init__(
        self,
        *,
        radius_bins_edges,
        dry=False,
        name=None,
        unit="kg^-1 m^-1",
    ):
        super().__init__(
            radius_bins_edges=radius_bins_edges,
            dry=dry,
            specific=True,
            name=name,
            unit=unit,
            stp=False,
        )


class ParticleSizeSpectrumPerVolume(ParticleSizeSpectrum):
    def __init__(
        self, *, radius_bins_edges, dry=False, name=None, unit="m^-3 m^-1", stp=False
    ):
        super().__init__(
            radius_bins_edges=radius_bins_edges,
            dry=dry,
            specific=False,
            name=name,
            unit=unit,
            stp=stp,
        )
