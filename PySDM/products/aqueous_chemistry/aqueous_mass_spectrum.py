"""
dry-radius-binned concentration of aqueous-chemistry relevant compounds (optionally
 expressed as specific concentration)
"""

import numpy as np
from chempy import Substance

from PySDM.dynamics.impl.chemistry_utils import AQUEOUS_COMPOUNDS
from PySDM.physics.constants import si
from PySDM.products.impl.spectrum_moment_product import SpectrumMomentProduct


class AqueousMassSpectrum(SpectrumMomentProduct):
    def __init__(
        self, *, key, dry_radius_bins_edges, specific=False, name=None, unit="kg/m^3"
    ):
        super().__init__(name=name, unit=unit, attr_unit="m")
        self.key = key
        self.dry_radius_bins_edges = dry_radius_bins_edges
        self.molar_mass = (
            Substance.from_formula(AQUEOUS_COMPOUNDS[key][0]).mass * si.g / si.mole
        )
        self.specific = specific

    def register(self, builder):
        builder.request_attribute("dry volume")
        builder.request_attribute(f"moles_{self.key}")

        dry_volume_bins_edges = builder.particulator.formulae.trivia.volume(
            self.dry_radius_bins_edges
        )
        self.attr_bins_edges = builder.particulator.backend.Storage.from_ndarray(
            dry_volume_bins_edges
        )

        super().register(builder)

        self.shape = (*builder.particulator.mesh.grid, len(self.attr_bins_edges) - 1)

    def _impl(self, **kwargs):
        vals = np.empty([self.particulator.mesh.n_cell, len(self.attr_bins_edges) - 1])
        self._recalculate_spectrum_moment(
            attr=f"moles_{self.key}", rank=1, filter_attr="dry volume"
        )

        for i in range(vals.shape[1]):
            self._download_spectrum_moment_to_buffer(rank=1, bin_number=i)
            vals[:, i] = self.buffer.ravel()
            self._download_spectrum_moment_to_buffer(rank=0, bin_number=i)
            vals[:, i] *= self.buffer.ravel()
        d_log10_diameter = np.diff(np.log10(2 * self.dry_radius_bins_edges))
        vals *= self.molar_mass / d_log10_diameter / self.particulator.mesh.dv

        if self.specific:
            self._download_to_buffer(self.particulator.environment["rhod"])
            vals[:] /= self.buffer
        return vals


class SpecificAqueousMassSpectrum(AqueousMassSpectrum):
    def __init__(self, key, dry_radius_bins_edges, name=None, unit="dimensionless"):
        super().__init__(
            key=key,
            dry_radius_bins_edges=dry_radius_bins_edges,
            name=name,
            unit=unit,
            specific=True,
        )
