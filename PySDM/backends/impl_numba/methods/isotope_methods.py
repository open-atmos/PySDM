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
            bolin_number,
            signed_water_mass,
            moles_heavy,
            molar_mass_heavy,
            ambient_isotope_mixing_ratio,
            cell_id,
            cell_volume,
            dry_air_density,
        ):
            # assume Bo = tau' / tau_total
            #           = (m'/dm') / (m/dm)
            #           = m'/m * dm/dm'
            # input (per droplet:
            #   Bo (discarding curvature/population effects)
            #   dm_total (actual - incl. population/curvature effects)
            # output:
            #   dm_heavy = dm_total / Bo * m'/m

            for sd_id in range(multiplicity.shape[0]):
                mass_ratio_heavy_to_total = (
                    moles_heavy[sd_id] * molar_mass_heavy
                ) / signed_water_mass[sd_id]
                dm_heavy = (
                    dm_total[sd_id] / bolin_number[sd_id] * mass_ratio_heavy_to_total
                )
                moles_heavy[sd_id] += dm_heavy / molar_mass_heavy
                mass_of_dry_air = (
                    dry_air_density[cell_id[sd_id]] * cell_volume
                )  # TODO: pass from outside
                ambient_isotope_mixing_ratio[cell_id[sd_id]] -= (
                    dm_heavy * multiplicity[sd_id] / mass_of_dry_air
                )

        return body

    def isotopic_fractionation(
        self,
        *,
        multiplicity,
        dm_total,
        bolin_number,
        signed_water_mass,
        moles_heavy,
        molar_mass_heavy,
        ambient_isotope_mixing_ratio,
        cell_id,
        cell_volume,
        dry_air_density,
    ):
        self._isotopic_fractionation_body(
            multiplicity=multiplicity.data,
            dm_total=dm_total.data,
            bolin_number=bolin_number.data,
            signed_water_mass=signed_water_mass.data,
            moles_heavy=moles_heavy.data,
            molar_mass_heavy=molar_mass_heavy,
            ambient_isotope_mixing_ratio=ambient_isotope_mixing_ratio.data,
            cell_id=cell_id.data,
            cell_volume=cell_volume,
            dry_air_density=dry_air_density.data,
        )

    @cached_property
    def _bolin_number_body(self):
        ff = self.formulae_flattened

        @numba.njit(**self.default_jit_flags)
        def body(output, molar_mass, cell_id, moles_heavy_isotope, relative_humidity):
            for i in numba.prange(output.shape[0]):  # pylint: disable=not-an-iterable
                output[i] = ff.isotope_relaxation_timescale__bolin_number(
                    moles_heavy_isotope=moles_heavy_isotope[i],
                    relative_humidity=relative_humidity[cell_id[i]],
                    molar_mass=molar_mass,
                )

        return body

    def bolin_number(
        self, *, output, molar_mass, cell_id, moles_heavy_isotope, relative_humidity
    ):
        self._bolin_number_body(
            output=output.data,
            molar_mass=molar_mass,
            cell_id=cell_id.data,
            moles_heavy_isotope=moles_heavy_isotope.data,
            relative_humidity=relative_humidity.data,
        )
