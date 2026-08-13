"""
CPU implementation of isotope-relates backend methods
"""

from functools import cached_property, lru_cache

import math
import numba
import numpy as np

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

        Mv = self.formulae.constants.Mv

        @numba.njit(**{**self.default_jit_flags, **{"parallel": False}})
        def body(
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
            water_vapour_mixing_ratio,
        ):  # pylint: disable=too-many-locals
            n_substeps = 1
            max_n_substeps = 256
            converged = False
            substep = 0
            new_molality = new_n_heavy_molecules = old_n_molecules = 0
            eps_1 = old_n_heavy_molecules = over_Bo = 1
            while n_substeps <= max_n_substeps:
                print("n_substeps=", n_substeps)
                delta_n_heavy_total = 0

                for sd_id in range(multiplicity.shape[0]):
                    if not converged:
                        print("   substep: ", substep)
                        old_n_molecules = (
                            signed_water_mass[sd_id] - dm_total[sd_id]
                        ) / Mv
                        old_n_heavy_molecules = moles_heavy_molecule[sd_id]
                        over_Bo = (
                            1 / bolin_number[sd_id]
                        )  # TODO check if updates Bo works and maybe it can be outside
                        eps_1 = (
                            molality_in_dry_air[cell_id[sd_id]]
                            / water_vapour_mixing_ratio[cell_id[sd_id]]
                        ) / (old_n_heavy_molecules / old_n_molecules)

                    dn_substep = dm_total[sd_id] / n_substeps / Mv
                    new_n_molecules = old_n_molecules + dn_substep

                    delta_n_heavy_per_drop = (
                        old_n_heavy_molecules / old_n_molecules * over_Bo * dn_substep
                    )
                    delta_n_heavy_total += delta_n_heavy_per_drop * multiplicity[sd_id]

                    eps_2 = (
                        new_molality / water_vapour_mixing_ratio[cell_id[sd_id]]
                    ) / (new_n_heavy_molecules / new_n_molecules)

                    if converged:
                        print(" converged: ", converged, "Bo=", over_Bo)
                        D_ratio = 1.01  # TODO TEMP!
                        rh = 0.9  # TODO TEMP!
                        Fk_over_Fd = 1  # TODO TEMP!
                        over_Bo += (
                            (eps_2 - eps_1)
                            * (D_ratio * rh * (1 + Fk_over_Fd))
                            / (rh - 1)
                        )
                        eps_1 = eps_2
                        moles_heavy_molecule[sd_id] += delta_n_heavy_per_drop
                        old_n_molecules = new_n_molecules

                mass_of_dry_air = dry_air_density[cell_id[0]] * cell_volume
                print(" molality=", molality_in_dry_air[cell_id[0]])
                new_molality = (
                    molality_in_dry_air[cell_id[0]]
                    - delta_n_heavy_total / mass_of_dry_air
                )
                print("  old_molality=", molality_in_dry_air[cell_id[0]])
                print("  new_molality=", new_molality, ", step=", substep)
                if not converged:
                    if new_molality < 0:  # check long vs short step
                        print("negative molality: ", new_molality, ", step=", substep)
                        n_substeps *= 2  # repeat substeps
                        if n_substeps > max_n_substeps:
                            break
                            assert (
                                False
                            ), "Exceeded maximum number of substeps, solution not found!"
                    else:
                        converged = True

                else:
                    molality_in_dry_air[cell_id[0]] = new_molality
                    print("  molality=", new_molality, ", step=", substep)
                    substep += 1
                if substep == n_substeps:
                    break

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
        water_vapour_mixing_ratio,
    ):  # pylint: disable=too-many-positional-arguments
        """Update heavy-isotope composition during droplet growth/evaporation."""
        self._isotopic_fractionation_body(
            cell_id=cell_id.data,
            cell_volume=cell_volume,
            multiplicity=multiplicity.data,
            dm_total=dm_total.data,
            signed_water_mass=signed_water_mass.data,
            dry_air_density=dry_air_density.data,
            molar_mass_heavy_molecule=molar_mass_heavy_molecule,
            moles_heavy_molecule=moles_heavy_molecule.data,
            bolin_number=bolin_number.data,
            molality_in_dry_air=molality_in_dry_air.data,
            water_vapour_mixing_ratio=water_vapour_mixing_ratio.data,
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
            *,
            output,
            cell_id,
            alpha,
            D_ratio,
            relative_humidity,
            temperature,
            water_vapour_mixing_ratio,
            moles_light_molecule,
            moles_heavy,
            molality_in_dry_air,
        ):  # pylint: disable=too-many-locals
            for i in numba.prange(output.shape[0]):  # pylint: disable=not-an-iterable
                T = temperature[cell_id[i]]
                pvs_water = ff.saturation_vapour_pressure__pvs_water(T)
                moles_heavy_atom = moles_heavy[i]
                moles_light_isotope = moles_light_molecule[i]  # TODO #1787

                isotopic_fraction = (
                    molality_in_dry_air[cell_id[i]]
                    / water_vapour_mixing_ratio[cell_id[i]]
                    * ff.constants.Mv
                )
                Fk = ff.drop_growth__Fk(T=T, K=ff.constants.K0, lv=ff.constants.l_tri)
                Fd = ff.drop_growth__Fd(
                    T=T,
                    D=ff.constants.D0,
                    pvs=pvs_water,
                )

                output[i] = ff.isotope_relaxation_timescale__bolin_number(
                    D_ratio_heavy_to_light=D_ratio(T),
                    alpha=alpha(T),
                    Fk=Fk,
                    Fd=Fd,
                    R_vap=ff.trivia__isotopic_ratio_assuming_single_heavy_isotope(
                        isotopic_fraction
                    ),
                    R_liq=moles_heavy_atom / moles_light_isotope,
                    relative_humidity=relative_humidity[cell_id[i]],
                )

        return body

    @lru_cache
    def alpha_l(self, isotope):
        return getattr(
            self.formulae_flattened,
            f"isotope_equilibrium_fractionation_factors__alpha_l_{isotope}",
        )

    @lru_cache
    def D_ratio(self, isotope):
        return getattr(
            self.formulae_flattened,
            f"isotope_diffusivity_ratios__ratio_{isotope}_heavy_to_light",
        )

    def bolin_number(
        self,
        *,
        output,
        cell_id,
        isotope,
        relative_humidity,
        temperature,
        moles_light_molecule,
        moles_heavy,
        molality_in_dry_air,
        water_vapour_mixing_ratio,
    ):
        """Bolin number per droplet"""
        self._bolin_number_body(
            output=output.data,
            cell_id=cell_id.data,
            D_ratio=self.D_ratio(isotope),
            alpha=self.alpha_l(isotope),
            relative_humidity=relative_humidity.data,
            temperature=temperature.data,
            moles_light_molecule=moles_light_molecule.data,
            moles_heavy=moles_heavy.data,
            molality_in_dry_air=molality_in_dry_air.data,
            water_vapour_mixing_ratio=water_vapour_mixing_ratio.data,
        )
