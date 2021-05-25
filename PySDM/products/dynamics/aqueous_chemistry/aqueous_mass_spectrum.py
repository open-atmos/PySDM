import numpy as np
from PySDM.physics.constants import convert_to, si
from PySDM.products.product import MomentProduct
from PySDM.physics.aqueous_chemistry.support import AQUEOUS_COMPOUNDS
from chempy import Substance


class AqueousMassSpectrum(MomentProduct):

    def __init__(self, key, dry_radius_bins_edges, specific=False):
        super().__init__(
            name=f'dm_{key}/dlog_10(dry diameter){"_spec" if specific else ""}',
            unit=f'µg / {"kg" if specific else "m3"} / (unit dD/D)',
            description=f'... {key} ...'
        )
        self.key = key
        self.moment_0 = None
        self.moments = None
        self.molar_mass = Substance.from_formula(AQUEOUS_COMPOUNDS[key][0]).mass * si.gram / si.mole
        self.dry_radius_bins_edges = dry_radius_bins_edges
        self.specific = specific

    def register(self, builder):
        super().register(builder)
        self.moment_0 = builder.core.backend.Storage.empty(1, dtype=int)
        self.moments = builder.core.backend.Storage.empty((1, 1), dtype=float)
        self.env = builder.core.env

    def get(self):
        volume_bins_edges = self.formulae.trivia.volume(self.dry_radius_bins_edges)
        vals = np.empty(len(volume_bins_edges) - 1)
        for i in range(len(vals)):
            self.download_moment_to_buffer(attr=f'moles_{self.key}', rank=1, filter_attr='dry volume',
                                           filter_range=(volume_bins_edges[i], volume_bins_edges[i + 1]))
            vals[i] = self.buffer[0]
            self.download_moment_to_buffer(attr='volume', rank=0, filter_attr='dry volume',
                                           filter_range=(volume_bins_edges[i], volume_bins_edges[i + 1]))
            vals[i] *= self.buffer[0]
        vals *= self.molar_mass / np.diff(np.log10(2 * self.dry_radius_bins_edges)) / self.core.mesh.dv
        convert_to(vals, si.ug)
        if self.specific:
            self.download_to_buffer(self.env['rhod'])
            vals[:] /= self.buffer
        return vals
