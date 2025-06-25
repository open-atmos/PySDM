"""
CPU implementation of isotope-relates backend methods
"""

from functools import cached_property

import numba

from PySDM.backends.impl_common.backend_methods import BackendMethods


class IsotopeMethods(BackendMethods):
    @cached_property
    def _isotopic_delta_body(self):
        ff = self.formulae_flattened

        @numba.njit(**self.default_jit_flags)
        def body(output, ratio, reference_ratio):
            for i in numba.prange(output.shape[0]):  # pylint: disable=not-an-iterable
                output[i] = ff.trivia__isotopic_ratio_2_delta(ratio[i], reference_ratio)

        return body

    def isotopic_delta(self, output, ratio, reference_ratio):
        self._isotopic_delta_body(output.data, ratio.data, reference_ratio)

    @cached_property
    def _isotopic_fractionation_body(self):
        @numba.njit(**self.default_jit_flags)
        def body(
            *,
            multiplicity,
            signed_timescale,
            moles,
            dt,
            ambient_isotope_mixing_ratio,
            cell_id,
            cell_volume,
            dry_air_density,
            molar_mass,
        ):
            for sd_id in numba.prange(multiplicity.shape[0]):
                dn = dt / signed_timescale[sd_id] * moles[sd_id]
                moles[sd_id] += dn
                mass_of_dry_air = (
                    dry_air_density[cell_id[sd_id]] * cell_volume[cell_id[sd_id]]
                )
                ambient_isotope_mixing_ratio[cell_id[sd_id]] -= (
                    dn * multiplicity[sd_id] * molar_mass / mass_of_dry_air
                )

        return body

    def isotopic_fractionation(
        self,
        *,
        isotope,
        multiplicity,
        signed_timescale,
        moles,
        dt,
        ambient_isotope_mixing_ratio,
        cell_id,
        cell_volume,
        dry_air_density,
    ):
        self._isotopic_fractionation_body(
            multiplicity=multiplicity.data,
            signed_timescale=signed_timescale.data,
            moles=moles.data,
            dt=dt,
            ambient_isotope_mixing_ratio=ambient_isotope_mixing_ratio.data,
            cell_id=cell_id.data,
            cell_volume=cell_volume.data,
            dry_air_density=dry_air_density.data,
            molar_mass=getattr(self.formulae.constants, f"M_{isotope}"),
        )
