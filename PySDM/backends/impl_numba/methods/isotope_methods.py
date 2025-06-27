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
        @numba.njit(**{**self.default_jit_flags, "parallel": False})
        def body(
            *,
            multiplicity,
            dm_total,
            bolins_number,
            moles_light,
            moles_heavy,
            molar_mass_light,
            molar_mass_heavy,
            ambient_isotope_mixing_ratio,
            cell_id,
            cell_volume,
            dry_air_density,
        ):
            # input:
            #   tau_heavy (ignoring population/curvature effects)
            #   tau_light (ditto!)
            #   dm_total (actual - incl. population/curvature effects)
            # output:
            #   dm_heavy = dm_total * tau_light / tau_heavy

            # tau' = m'/dm' * dt
            # tau = m/dm * dt
            # dm'/dm = tau/tau' * m'/m

            # dm' = dm * tau/tau' * m'/m
            #            ^^^^^^^
            #            Bolin's 1/c1

            for sd_id in range(multiplicity.shape[0]):
                mass_ratio_heavy_to_light = (moles_heavy[sd_id] * molar_mass_heavy) / (
                    moles_light[sd_id] * molar_mass_light
                )
                dm_heavy_approx = (
                    dm_total[sd_id] / bolins_number * mass_ratio_heavy_to_light
                )
                moles_heavy[sd_id] += dm_heavy_approx / molar_mass_heavy
                mass_of_dry_air = (
                    dry_air_density[cell_id[sd_id]] * cell_volume[cell_id[sd_id]]
                )
                ambient_isotope_mixing_ratio[cell_id[sd_id]] -= (
                    dm_heavy_approx * multiplicity[sd_id] / mass_of_dry_air
                )

        return body

    def isotopic_fractionation(
        self,
        *,
        molar_mass,
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
            molar_mass=molar_mass,
        )

    @cached_property
    def _bolins_number_body(self):
        ff = self.formulae_flattened

        @numba.njit(**self.default_jit_flags)
        def body(output, molar_mass, cell_id, moles_heavy_isotope, relative_humidity):
            for i in numba.prange(output.shape[0]):  # pylint: disable=not-an-iterable
                output[i] = ff.isotope_relaxation_timescale__bolins_number(
                    moles_heavy_isotope=moles_heavy_isotope[i],
                    relative_humidity=relative_humidity[cell_id[i]],
                    molar_mass=molar_mass,
                )

        return body

    def bolins_number(
        self, *, output, molar_mass, cell_id, moles_heavy_isotope, relative_humidity
    ):
        self._bolins_number_body(
            output=output.data,
            molar_mass=molar_mass,
            cell_id=cell_id.data,
            moles_heavy_isotope=moles_heavy_isotope.data,
            relative_humidity=relative_humidity.data,
        )
