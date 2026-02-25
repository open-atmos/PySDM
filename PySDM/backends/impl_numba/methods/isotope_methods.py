"""
CPU implementation of isotope-relates backend methods
"""

from functools import cached_property

import numba

from PySDM.backends.impl_common.backend_methods import BackendMethods


class IsotopeMethods(BackendMethods):
    """
    CPU backend methods for droplet isotope processing.

    Provides Numba-accelerated kernels for:
    - isotopic ratio to delta conversion,
    - Bolin number evaluation,
    - heavy-isotope fractionation during condensation/evaporation.
    """

    @cached_property
    def _isotopic_delta_body(self):
        """Numba kernel to convert isotopic ratios to delta values."""
        ff = self.formulae_flattened

        @numba.njit(**self.default_jit_flags)
        def body(output, ratio, reference_ratio):
            for i in numba.prange(output.shape[0]):  # pylint: disable=not-an-iterable
                output[i] = ff.trivia__isotopic_ratio_2_delta(ratio[i], reference_ratio)

        return body

    def isotopic_delta(self, output, ratio, reference_ratio):
        """Compute isotopic delta for droplets."""
        self._isotopic_delta_body(output.data, ratio.data, reference_ratio)

    @cached_property
    def _isotopic_fractionation_body(self):
        """
        Kernel updating heavy-isotope content during phase change.

        Computes the heavy-isotope mass change per droplet using the Bolin
        number (Bo) and the instantaneous heavy-to-total mass ratio:

            dm_heavy = (dm_total / Bo) * (m_heavy / m_total)
            dn_heavy = dm_heavy / M_heavy

        Updates:
        - moles_heavy_molecule per droplet,
        - molality of heavy isotope in dry air.
        """

        @numba.njit(**{**self.default_jit_flags, **{"parallel": False}})
        def body(
            cell_id,
            cell_volume,
            multiplicity,
            dm_total,
            signed_water_mass,
            dry_air_density,
            molar_mass_heavy_molecule,
            moles_heavy_molecule,
            bolin_number,
            molality_in_dry_air,
        ):
            for sd_id in range(multiplicity.shape[0]):
                mass_ratio_heavy_to_total = (
                    moles_heavy_molecule[sd_id] * molar_mass_heavy_molecule
                ) / signed_water_mass[sd_id]
                if bolin_number[sd_id] == 0:
                    dm_heavy = 0
                else:
                    dm_heavy = (
                        dm_total[sd_id]
                        / bolin_number[sd_id]
                        * mass_ratio_heavy_to_total
                    )
                dn_heavy_molecule = dm_heavy / molar_mass_heavy_molecule
                moles_heavy_molecule[sd_id] += dn_heavy_molecule
                mass_of_dry_air = (
                    dry_air_density[cell_id[sd_id]] * cell_volume[cell_id[sd_id]]
                )
                molality_in_dry_air[cell_id[sd_id]] -= (
                    dn_heavy_molecule * multiplicity[sd_id] / mass_of_dry_air
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
        molar_mass_heavy_molecule,
        moles_heavy_molecule,
        bolin_number,
        molality_in_dry_air,
    ):  # pylint: disable=too-many-positional-arguments
        """Update heavy-isotope composition during droplet growth/evaporation."""
        self._isotopic_fractionation_body(
            cell_id.data,
            cell_volume,
            multiplicity.data,
            dm_total.data,
            signed_water_mass.data,
            dry_air_density.data,
            molar_mass_heavy_molecule,
            moles_heavy_molecule.data,
            bolin_number.data,
            molality_in_dry_air.data,
        )

    @cached_property
    def _bolin_number_body(self):
        """
        Kernel computing the Bolin number per droplet.

        Calculates the isotope relaxation timescale ratio (Bolin number)
        based on droplet composition, ambient temperature, relative humidity,
        and water vapor properties:

            Bo = tau_heavy / tau_bulk

        Updates the output array with per-droplet Bolin numbers.
        """
        ff = self.formulae_flattened

        @numba.njit(**{**self.default_jit_flags, **{"parallel": False}})
        def body(
            output,
            cell_id,
            relative_humidity,
            temperature,
            density_dry_air,
            moles_light_molecule,
            moles_heavy,
            molality_in_dry_air,
        ):  # pylint: disable=too-many-locals,too-many-positional-arguments

            for i in numba.prange(output.shape[0]):  # pylint: disable=not-an-iterable
                T = temperature[cell_id[i]]
                pvs_water = ff.saturation_vapour_pressure__pvs_water(T)
                moles_heavy_atom = moles_heavy[i]
                moles_light_isotope = moles_light_molecule[i]  # TODO #1787
                conc_vap_total = (
                    pvs_water * relative_humidity[cell_id[i]] / ff.constants.R_str / T
                )
                rho_v = pvs_water / T / ff.constants.Rv

                isotopic_fraction = ff.trivia__isotopic_fraction(
                    molality_in_dry_air=molality_in_dry_air[cell_id[i]],
                    density_dry_air=density_dry_air[cell_id[i]],
                    total_vap_concentration=conc_vap_total,
                )
                D_ratio_heavy_to_light = (
                    ff.isotope_diffusivity_ratios__ratio_2H_heavy_to_light(T)
                )
                output[i] = ff.isotope_relaxation_timescale__bolin_number(
                    D_ratio_heavy_to_light=D_ratio_heavy_to_light,
                    alpha=ff.isotope_equilibrium_fractionation_factors__alpha_l_2H(T),
                    D_light=ff.constants.D0,
                    Fk=ff.drop_growth__Fk(
                        T=T, K=ff.constants.K0, lv=ff.constants.l_tri
                    ),
                    R_vap=ff.trivia__isotopic_ratio_assuming_single_heavy_isotope(
                        isotopic_fraction
                    ),
                    R_liq=moles_heavy_atom / moles_light_isotope,
                    relative_humidity=relative_humidity[cell_id[i]],
                    rho_v=rho_v,
                )

        return body

    def bolin_number(
        self,
        *,
        output,
        cell_id,
        relative_humidity,
        temperature,
        density_dry_air,
        moles_light_molecule,
        moles_heavy,
        molality_in_dry_air,
    ):
        """Bolin number per droplet"""
        self._bolin_number_body(
            output.data,
            cell_id.data,
            relative_humidity.data,
            temperature.data,
            density_dry_air.data,
            moles_light_molecule.data,
            moles_heavy.data,
            molality_in_dry_air.data,
        )
