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
            cell_id,
            cell_volume,
            multiplicity,
            dm_total,
            signed_water_mass,
            dry_air_density,
            molar_mass_heavy,
            moles_heavy,
            delta_heavy,
            bolin_number,
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
                    dry_air_density[sd_id] * cell_volume
                )  # TODO: pass from outside
                delta_heavy[cell_id[sd_id]] -= (  # FIXME ambient isotope mixing ratio
                    dm_heavy * multiplicity[sd_id] / mass_of_dry_air
                )

        return body

    def isotopic_fractionation(
        self,
        *,
        cell_id,
        cell_volume,
        multiplicity,
        dm_total,
        signed_water_mass,
        dry_air_density,
        molar_mass_heavy,
        moles_heavy,
        delta_heavy,
        bolin_number,
    ):
        self._isotopic_fractionation_body(
            cell_id=cell_id.data,
            cell_volume=cell_volume,
            multiplicity=multiplicity.data,
            dm_total=dm_total.data,
            signed_water_mass=signed_water_mass.data,
            dry_air_density=dry_air_density.data,
            molar_mass_heavy=molar_mass_heavy,
            moles_heavy=moles_heavy.data,
            delta_heavy=delta_heavy.data,
            bolin_number=bolin_number.data,
        )

    @cached_property
    def _bolin_number_body(self):
        ff = self.formulae_flattened

        @numba.njit(**self.default_jit_flags)
        def body(
            output,
            cell_id,
            relative_humidity,
            temperature,
            moles_light_water,
            moles_heavy,
            delta_heavy,
        ):
            for i in numba.prange(output.shape[0]):  # pylint: disable=not-an-iterable
                moles_heavy_isotope = moles_heavy[i]
                moles_light_isotope = moles_light_water[i]
                output[i] = ff.isotope_relaxation_timescale__bolin_number(
                    D_ratio_heavy_to_light=ff.isotope_diffusivity_ratios__ratio_2H_heavy_to_light(
                        temperature[cell_id[i]]
                    ),
                    alpha=ff.isotope_equilibrium_fractionation_factors__alpha_l_2H(
                        temperature[cell_id[i]]
                    ),
                    D_light=1e-5,  # ff.constants__DO,
                    Fk_Howell=1,  # TODO
                    R_vap=delta_heavy[cell_id[i]],  # TODO
                    R_liq=moles_heavy_isotope / moles_light_isotope,
                    relative_humidity=relative_humidity[cell_id[i]],
                )

        return body

    def bolin_number(
        self,
        *,
        output,
        cell_id,
        relative_humidity,
        temperature,
        moles_light_water,
        moles_heavy,
        delta_heavy,
    ):
        self._bolin_number_body(
            output=output.data,
            cell_id=cell_id.data,
            relative_humidity=relative_humidity.data,
            temperature=temperature.data,
            moles_light_water=moles_light_water.data,
            moles_heavy=moles_heavy.data,
            delta_heavy=delta_heavy.data,
        )
