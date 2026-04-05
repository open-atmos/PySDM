import numpy as np

from PySDM import Builder
from PySDM.environments import Box
from PySDM.backends import CPU
from PySDM.dynamics import Condensation, IsotopicFractionation
from PySDM.dynamics.isotopic_fractionation import HEAVY_ISOTOPES


class Simulation:
    @staticmethod
    def make_particulator(
        *,
        ff,
        molecular_isotopic_ratio,
        initial_R_vap,
        attributes,
        n_sd,
        dv,
        dt,
        relative_humidity,
        T,
        isotope,
    ):  # pylint: disable=too-many-arguments
        """needed setup for simulation"""
        const = ff.constants
        if isotope == "2H":
            molar_mass_heavy = const.M_2H_1H_16O
            atoms_per_heavy_molecule = 1
        elif isotope[-1] == "O":
            molar_mass_heavy = getattr(const, f"M_1H2_{isotope}")
            atoms_per_heavy_molecule = 1
        else:
            molar_mass_heavy = 0
            atoms_per_heavy_molecule = 0
        attributes[f"moles_{isotope}"] = ff.trivia.moles_heavy_atom(
            mass_total=attributes["signed water mass"],
            mass_other_heavy_isotopes=0,
            molar_mass_light_molecule=const.M_1H2_16O,
            molar_mass_heavy_molecule=molar_mass_heavy,
            molecular_isotopic_ratio=molecular_isotopic_ratio,
            atoms_per_heavy_molecule=atoms_per_heavy_molecule,
        )
        builder = Builder(
            n_sd=n_sd,
            backend=CPU(
                formulae=ff,
            ),
            environment=Box(dv=dv, dt=dt),
        )
        builder.add_dynamic(Condensation())
        builder.add_dynamic(IsotopicFractionation(isotopes=(isotope,)))

        rho_d = const.p_STP / const.Rd / T
        builder.particulator.environment["dry_air_density"] = rho_d
        builder.particulator.environment["RH"] = relative_humidity
        builder.particulator.environment["T"] = T
        initial_conc_vap = (
            ff.saturation_vapour_pressure.pvs_water(T)
            * relative_humidity
            / const.R_str
            / T
        )

        molality_in_dry_air = ff.trivia.molality_in_dry_air(
            isotopic_fraction=ff.trivia.isotopic_fraction_assuming_single_heavy_isotope(
                isotopic_ratio=initial_R_vap
            ),
            density_dry_air=rho_d,
            total_vap_concentration=initial_conc_vap,
        )

        for iso in HEAVY_ISOTOPES:
            if iso == isotope:
                builder.particulator.environment[f"molality {iso} in dry air"] = (
                    molality_in_dry_air
                )
            else:
                builder.particulator.environment[f"molality {iso} in dry air"] = 0
        builder.request_attribute(f"delta_{isotope}")
        builder.request_attribute(f"Bolin number for {isotope}")
        return builder.build(attributes=attributes, products=()), initial_conc_vap

    @staticmethod
    def do_one_step(
        *,
        ff,
        particulator,
        dm_dt_per_droplet,
        init_m_R_liq,
        initial_conc_vap,
        isotope,
    ):
        """perform one step of simulation"""
        if isotope[-1] == "O":
            light_isotope = "16O"
            atoms_per_light_molecule = 1
        elif isotope == "2H":
            light_isotope = "1H"
            atoms_per_light_molecule = 2

        isotopic_fraction = ff.trivia.isotopic_fraction(
            particulator.environment[f"molality {isotope} in dry air"][0],
            particulator.environment["dry_air_density"][0],
            initial_conc_vap,
        )
        R_vap = ff.trivia.isotopic_ratio_assuming_single_heavy_isotope(
            isotopic_fraction=isotopic_fraction
        )

        initial_R_liq = (
            particulator.attributes[f"moles_{isotope}"][0]
            / particulator.attributes[f"moles_{light_isotope}"][0]
        )

        np.testing.assert_approx_equal(
            initial_R_liq, init_m_R_liq / atoms_per_light_molecule, significant=4
        )
        init_Bo = particulator.attributes[f"Bolin number for {isotope}"][0]

        particulator.attributes["diffusional growth mass change"].data[:] = (
            dm_dt_per_droplet * particulator.dt
        )
        particulator.dynamics["IsotopicFractionation"]()

        # FIXME
        # total_vap_conc = n/V
        # molality = n'/m_d
        # isotopic_conc  = n'/n
        # wvmr = m_vap/m_d = n_vap * Mv / m_d
        #       => n_vap = wvmr * m_d / Mv
        # R_vap = n'_vap / n_vap = molality * m_d / n_vap
        #       = molality * m_d / wvmr / m_d * Mv
        #       = molality / wvmr * Mv

        total_vap_conc = (
            initial_conc_vap
            - dm_dt_per_droplet
            * particulator.dt
            * particulator.attributes["multiplicity"][0]
            / ff.constants.Mv
            / particulator.environment.mesh.dv[0]
        )
        new_isotopic_fraction = ff.trivia.isotopic_fraction(
            particulator.environment[f"molality {isotope} in dry air"][0],
            particulator.environment["dry_air_density"][0],
            total_vap_conc,
        )

        new_R_vap = ff.trivia.isotopic_ratio_assuming_single_heavy_isotope(
            isotopic_fraction=new_isotopic_fraction
        )
        new_R_liq = (
            particulator.attributes[f"moles_{isotope}"][0]
            / particulator.attributes[f"moles_{light_isotope}"][0]
        )
        dR_dt_vap = (new_R_vap - R_vap) / particulator.dt
        dR_dt_liq = (new_R_liq - initial_R_liq) / particulator.dt
        return dR_dt_vap, dR_dt_liq, init_Bo
