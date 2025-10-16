"""
unit test for the IsotopicFractionation dynamic
"""

from contextlib import nullcontext

import numpy as np
import pytest

from PySDM import Builder, Formulae
from PySDM.dynamics import Condensation, IsotopicFractionation
from PySDM.dynamics.isotopic_fractionation import HEAVY_ISOTOPES
from PySDM.environments import Box
from PySDM.physics import si
from PySDM.physics.constants_defaults import PER_MILLE, VSMOW_R_2H

DUMMY_ATTRIBUTES = {
    attr: np.asarray([0 if attr == "multiplicity" else np.nan])
    for attr in (
        "multiplicity",
        "signed water mass",
        "dry volume",
        "kappa times dry volume",
        *[f"moles_{isotope}" for isotope in HEAVY_ISOTOPES],
    )
}


def isotope_settings_factory(
    request_attributes=None,
    initial_values_for_attributes=None,
    additional_parameters=None,
):
    formulae = Formulae(  # TODO
        isotope_relaxation_timescale="GedzelmanAndArnold1994",
        isotope_diffusivity_ratios="GrahamsLaw",
        isotope_equilibrium_fractionation_factors="VanHook1968",
    )
    RH = 1
    multiplicity = np.ones(1)
    m_t = 1 * si.kg

    temperature = formulae.trivia.C2K(10) * si.K
    dt = -1 * si.s
    if additional_parameters is not None:
        cell_volume = additional_parameters["cell_volume"]
    else:
        cell_volume = 1 * si.m**3
    mass_dry_air = 1 * si.ng  # Question: update n_light?, dividing by mass_dry_air
    attributes = DUMMY_ATTRIBUTES.copy()
    attributes["moles_2H"] = 1 * si.moles
    for isotope in HEAVY_ISOTOPES:
        if isotope != "2H":
            attributes[f"moles_{isotope}"] = 0 * si.moles
    attributes["signed water mass"] = m_t
    attributes["multiplicity"] = multiplicity
    if initial_values_for_attributes is not None:
        attributes.update(initial_values_for_attributes)
    builder = Builder(
        n_sd=1,
        backend=backend_class(
            formulae=formulae,
        ),
        environment=Box(dv=cell_volume, dt=dt),
    )
    builder.add_dynamic(Condensation())
    builder.add_dynamic(IsotopicFractionation(isotopes=HEAVY_ISOTOPES))
    if request_attributes:
        for attribute in request_attributes:
            builder.request_attribute(attribute)

    particulator = builder.build(attributes=attributes, products=())
    particulator.environment["T"] = temperature
    particulator.environment["RH"] = RH
    particulator.environment["dry_air_density"] = mass_dry_air / cell_volume
    particulator.environment["molar mixing ratio 2H"] = 1
    for isotope in HEAVY_ISOTOPES:
        if isotope != "2H":
            particulator.environment[f"molar mixing ratio {isotope}"] = 0  #

    return formulae, particulator


class TestIsotopicFractionation:
    @staticmethod
    @pytest.mark.parametrize(
        "dynamics, context",
        (
            pytest.param(
                (Condensation(), IsotopicFractionation(isotopes=("1H",))), nullcontext()
            ),
            pytest.param(
                (IsotopicFractionation(isotopes=("1H",)),),
                pytest.raises(AssertionError, match="dynamics"),
            ),
            pytest.param(
                (IsotopicFractionation(isotopes=("1H",)), Condensation()),
                pytest.raises(AssertionError, match="dynamics"),
            ),
        ),
    )
    def test_ensure_condensation_executed_before(backend_instance, dynamics, context):
        # arrange
        builder = Builder(
            n_sd=1, backend=backend_instance, environment=Box(dv=np.nan, dt=1 * si.s)
        )
        for dynamic in dynamics:
            builder.add_dynamic(dynamic)

        # act
        with context:
            builder.build(attributes=DUMMY_ATTRIBUTES)

    @staticmethod
    def test_call_marks_all_isotopes_as_updated(isotope_settings_factory):
        # arrange
        isotope_settings = isotope_settings_factory()

        _, particulator = isotope_settings

        for isotope in HEAVY_ISOTOPES:
            # pylint:disable=protected-access
            assert (
                particulator.attributes._ParticleAttributes__attributes[
                    f"moles_{isotope}"
                ].timestamp
                == particulator.attributes._ParticleAttributes__attributes[
                    "multiplicity"
                ].timestamp
            )

        # act
        particulator.dynamics["IsotopicFractionation"]()

        # assert
        for isotope in HEAVY_ISOTOPES:
            # pylint:disable=protected-access
            assert (
                particulator.attributes._ParticleAttributes__attributes[
                    f"moles_{isotope}"
                ].timestamp
                > particulator.attributes._ParticleAttributes__attributes[
                    "multiplicity"
                ].timestamp
            )

    @staticmethod
    def test_no_isotope_fractionation_if_droplet_size_unchanged(
        # isotope_settings_factory,
    ):
        """neither a bug nor a feature :) - just a simplification (?)"""
        # arrange
        initial_values = {"moles_2H": 44.0}
        isotope_settings = isotope_settings_factory(
            initial_values_for_attributes=initial_values,
        )
        _, particulator = isotope_settings

        # act
        particulator.attributes["diffusional growth mass change"].data[:] = 0
        particulator.dynamics[
            "IsotopicFractionation"
        ]()  # TODO: call condensation as well!

        # assert
        assert particulator.attributes["moles_2H"][0] == initial_values["moles_2H"]

    @staticmethod
    @pytest.mark.parametrize("molecular_R_rain", np.linspace(0.8, 1, 5) * VSMOW_R_2H)
    def test_initial_condition_for_delta_isotopes(
        backend_instance,
        molecular_R_rain,
    ):
        # arrange
        formulae = Formulae()
        const = formulae.constants
        temperature = formulae.trivia.C2K(10) * si.K
        dt = -1 * si.s
        cell_volume = 1 * si.m**3
        mass_dry_air = 1 * si.ng
        m_t = 1 * si.kg

        attributes = DUMMY_ATTRIBUTES.copy()
        moles_total = m_t / const.Mv

        for isotope in HEAVY_ISOTOPES:
            attributes[f"moles_{isotope}"] = 0 * si.mol

        heavy_moles_sum_without_2H = sum(
            attributes[f"moles_{isotope}"] for isotope in HEAVY_ISOTOPES
        )
        moles_2H = (
            molecular_R_rain
            * (moles_total - heavy_moles_sum_without_2H)
            / (1 + molecular_R_rain)
        )
        attributes["moles_2H"] = moles_2H
        attributes["signed water mass"] = m_t
        attributes["multiplicity"] = np.ones(1)
        builder = Builder(
            n_sd=1,
            backend=backend_instance,
            environment=Box(dv=cell_volume, dt=dt),
        )
        builder.add_dynamic(Condensation())
        builder.add_dynamic(IsotopicFractionation(isotopes=HEAVY_ISOTOPES))
        builder.request_attribute("delta_2H")

        particulator = builder.build(attributes=attributes, products=())
        particulator.environment["T"] = temperature
        particulator.environment["RH"] = 1
        particulator.environment["dry_air_density"] = mass_dry_air / cell_volume
        for isotope in HEAVY_ISOTOPES:
            # if isotope != "2H":
            particulator.environment[f"molar mixing ratio {isotope}"] = 0

        R_rain = (
            particulator.attributes["moles_2H"][0]
            / particulator.attributes["moles_1H"][0]
        )
        delta_rain = formulae.trivia.isotopic_ratio_2_delta(R_rain, const.VSMOW_R_2H)

        # assert
        np.testing.assert_approx_equal(
            particulator.attributes["moles_2H"][0],
            moles_2H,
            significant=5,
        )
        np.testing.assert_approx_equal(
            particulator.attributes["delta_2H"][0],
            delta_rain,
        )

    @staticmethod
    @pytest.mark.parametrize(
        "RH, delta_rain, sign_of_dR_vap, sign_of_dR_rain",
        (
            (0.5, 0.86 - 1, -1, 1),
            (0.5, 0.9 - 1, 1, 1),
            (0.5, 0.98 - 1, 1, -1),
        ),
    )
    def test_scenario_from_gedzelman_fig_2_for_single_superdroplet_and_single_isotope(
        RH,
        delta_rain,
        sign_of_dR_vap,
        sign_of_dR_rain,
        backend_class,
    ):
        # arrange
        formulae = Formulae(  # TODO
            isotope_relaxation_timescale="GedzelmanAndArnold1994",
            isotope_diffusivity_ratios="GrahamsLaw",
            isotope_equilibrium_fractionation_factors="VanHook1968",
        )
        const = formulae.constants
        cell_volume = 1 * si.m**3
        multiplicity = np.ones(1)
        m_t = 1 * si.kg
        temperature = formulae.trivia.C2K(10) * si.K
        dt = -1 * si.s
        mass_dry_air = 1 * si.ng

        # initial_R_vap = formulae.trivia.isotopic_delta_2_ratio(
        #     -200 * PER_MILLE, const.VSMOW_R_2H
        # )
        n_vap_total = (
            RH
            * formulae.saturation_vapour_pressure.pvs_water(temperature)
            * cell_volume
            / const.R_str
            / temperature
        )

        attributes = DUMMY_ATTRIBUTES.copy()
        attributes["moles_2H"] = formulae.trivia.moles_heavy_atom(  # CHECK
            delta=delta_rain,
            mass_total=m_t,
            molar_mass_heavy_molecule=const.M_2H_1H_16O,
            reference_ratio=const.VSMOW_R_2H,
            light_atoms_per_light_molecule=2,
        )
        for isotope in HEAVY_ISOTOPES:
            if isotope != "2H":
                attributes[f"moles_{isotope}"] = 0 * si.moles

        attributes["signed water mass"] = m_t
        attributes["multiplicity"] = multiplicity
        builder = Builder(
            n_sd=1,
            backend=backend_class(
                formulae=formulae,
            ),
            environment=Box(dv=cell_volume, dt=dt),
        )
        builder.add_dynamic(Condensation())
        builder.add_dynamic(IsotopicFractionation(isotopes=HEAVY_ISOTOPES))
        builder.request_attribute("delta_2H")

        particulator = builder.build(attributes=attributes, products=())
        particulator.environment["T"] = temperature
        particulator.environment["RH"] = RH
        particulator.environment["dry_air_density"] = mass_dry_air / cell_volume
        # particulator.environment["molar mixing ratio 2H"] = (
        #     formulae.trivia.R_vap_to_molar_mixing_ratio_assuming_single_heavy_isotope(
        #         R_vap=initial_R_vap,
        #         n_vap_total=n_vap_total,
        #         mass_dry_air=mass_dry_air,
        #     )
        # )
        for isotope in HEAVY_ISOTOPES:
            if isotope != "2H":
                particulator.environment[f"molar mixing ratio {isotope}"] = 0
                np.testing.assert_equal(
                    particulator.attributes[f"moles_{isotope}"][0], 0
                )

        # sanity check for initial condition
        np.testing.assert_approx_equal(
            particulator.attributes["delta_2H"][0],
            delta_rain,
            significant=5,
        )

        # act
        droplet_dm = -d_mixing_ratio_env * mass_dry_air / multiplicity
        particulator.attributes["diffusional growth mass change"].data[:] = droplet_dm
        particulator.dynamics["IsotopicFractionation"]()

        n_vap_total = formulae.trivia.n_vap_total(
            RH=particulator.environment["RH"].data,
            temperature=temperature,
            pvs_water=formulae.saturation_vapour_pressure.pvs_water(temperature),
            cell_volume=cell_volume,
        )
        new_R_vap = (
            formulae.trivia.molar_mixing_ratio_to_R_vap_assuming_single_heavy_isotope(
                molar_mixing_ratio=particulator.environment[
                    "molar mixing ratio 2H"
                ].data[0],
                n_vap_total=n_vap_total,
                mass_dry_air=particulator.environment["dry_air_density"].data[0]
                * cell_volume,
            )
        )
        new_delta_rain = particulator.attributes["delta_2H"][0]

        # assert
        assert np.sign(new_R_vap - R_vap) == sign_of_dR_vap
        assert np.sign(new_delta_rain - delta_rain) == sign_of_dR_rain

    # TODO
    @staticmethod
    def test_heavy_isotope_changes_match_bolin_number():
        pass
