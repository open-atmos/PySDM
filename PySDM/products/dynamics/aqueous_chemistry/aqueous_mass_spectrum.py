import numpy as np
from PySDM.physics.constants import convert_to, si
from PySDM.products.product import SpectrumMomentProduct
from PySDM.physics.aqueous_chemistry.support import AQUEOUS_COMPOUNDS
from chempy import Substance


class AqueousMassSpectrum(SpectrumMomentProduct):

    def __init__(self, key, dry_radius_bins_edges, specific=False):
        super().__init__(
            name=f'dm_{key}/dlog_10(dry diameter){"_spec" if specific else ""}',
            unit=f'µg / {"kg" if specific else "m3"} / (unit dD/D)',
            description=f'... {key} ...'
        )
        self.key = key
        self.dry_radius_bins_edges = dry_radius_bins_edges
        self.molar_mass = Substance.from_formula(AQUEOUS_COMPOUNDS[key][0]).mass * si.gram / si.mole
        self.specific = specific

    def register(self, builder):
        builder.request_attribute('dry volume')
        builder.request_attribute(f'moles_{self.key}')

        dry_volume_bins_edges = builder.core.formulae.trivia.volume(self.dry_radius_bins_edges)
        self.attr_bins_edges = builder.core.bck.Storage.from_ndarray(dry_volume_bins_edges)

        super().register(builder)

        self.shape = (*builder.core.mesh.grid, len(self.attr_bins_edges) - 1)

    def get(self):
        vals = np.empty([self.core.mesh.n_cell, len(self.attr_bins_edges) - 1])
        self.recalculate_spectrum_moment(attr=f'moles_{self.key}', rank=1, filter_attr='dry volume')

        for i in range(vals.shape[1]):
            self.download_spectrum_moment_to_buffer(rank=1, bin_number=i)
            vals[:, i] = self.buffer.ravel()
            self.download_spectrum_moment_to_buffer(rank=0, bin_number=i)
            vals[:, i] *= self.buffer.ravel()

        vals *= self.molar_mass / np.diff(np.log10(2 * self.dry_radius_bins_edges)) / self.core.mesh.dv
        convert_to(vals, si.ug)

        if self.specific:
            self.download_to_buffer(self.core.environment['rhod'])
            vals[:] /= self.buffer
        return vals
