"""
unit tests for isotope-related attributes
"""

import numpy as np
import pytest

from PySDM import Builder, Formulae
from PySDM.dynamics.isotopic_fractionation import HEAVY_ISOTOPES
from PySDM.environments import Box
from PySDM.physics import constants_defaults, si
from PySDM.physics.trivia import Trivia


def dummy_attrs(length):
    return {
        "signed water mass": np.asarray([0.666 * si.g] * length),
        "multiplicity": np.asarray([-1] * length, dtype=int),
    }


class TestIsotopes:
    @staticmethod
    @pytest.mark.parametrize("isotope", HEAVY_ISOTOPES)
    def test_heavy_isotope_moles_attributes(backend_instance, isotope):
        # arrange
        values = [1, 2, 3]
        builder = Builder(
            n_sd=len(values),
            backend=backend_instance,
            environment=Box(dt=np.nan, dv=np.nan),
        )
        particulator = builder.build(
            attributes={
                f"moles_{isotope}": np.asarray(values),
                **dummy_attrs(len(values)),
            }
        )

        # act
        attr_values = particulator.attributes[f"moles_{isotope}"].to_ndarray()

        # assert
        np.testing.assert_array_equal(attr_values, values)

    @staticmethod
    @pytest.mark.parametrize("isotope", HEAVY_ISOTOPES)
    @pytest.mark.parametrize(
        "heavier_water_specific_content",
        (
            0.00001,
            0.0001,
        ),
    )
    def test_delta_attribute(backend_class, isotope, heavier_water_specific_content):
        # arrange
        builder = Builder(
            n_sd=1,
            backend=backend_class(double_precision=True),
            environment=Box(dt=np.nan, dv=np.nan),
        )
        builder.request_attribute(f"delta_{isotope}")

        attributes = {
            **dummy_attrs(1),
            **{f"moles_{k}": np.zeros(1) for k in HEAVY_ISOTOPES if k != isotope},
        }
        heavier_water_molar_mass = {
            "2H": constants_defaults.M_16O
            + constants_defaults.M_1H
            + constants_defaults.M_2H,
            "3H": constants_defaults.M_16O
            + constants_defaults.M_1H
            + constants_defaults.M_3H,
            "17O": constants_defaults.M_17O + 2 * constants_defaults.M_1H,
            "18O": constants_defaults.M_18O + 2 * constants_defaults.M_1H,
        }[isotope]
        attributes[f"moles_{isotope}"] = np.asarray(
            [
                heavier_water_specific_content
                * attributes["signed water mass"]
                / heavier_water_molar_mass
            ]
        )
        particulator = builder.build(attributes=attributes)

        # act
        (delta,) = particulator.attributes[f"delta_{isotope}"].to_ndarray()

        # assert
        ((n_heavy_isotope,),) = attributes[f"moles_{isotope}"]
        (n_light_water,) = (
            (1 - heavier_water_specific_content)
            * attributes["signed water mass"]
            / (constants_defaults.M_1H * 2 + constants_defaults.M_16O)
        )
        if isotope[-1] == "O":
            n_light_isotope = n_light_water
        elif isotope[-1] == "H":
            n_light_isotope = n_light_water * 2 + n_heavy_isotope
        else:
            raise NotImplementedError()
        np.testing.assert_approx_equal(
            actual=delta,
            desired=Trivia.isotopic_ratio_2_delta(
                reference_ratio=getattr(constants_defaults, f"VSMOW_R_{isotope}"),
                ratio=n_heavy_isotope / n_light_isotope,
            ),
            significant=5,
        )

    @staticmethod
    @pytest.mark.parametrize("heavy_isotope", HEAVY_ISOTOPES)
    @pytest.mark.parametrize(
        "moles_heavy, relative_humidity, expected_sign_of_tau",
        (
            (1, 0.99, -1),
            (1, 1.01, 1),
        ),
    )
    @pytest.mark.parametrize("variant", ("ZabaEtAl",))
    def test_bolin_number_attribute(
        backend_class,
        heavy_isotope: str,
        moles_heavy: float,
        relative_humidity: float,
        expected_sign_of_tau: float,
        variant: str,
    ):  # pylint: disable=too-many-arguments
        if backend_class.__name__ != "Numba":
            pytest.skip("# TODO #1787 - isotopes on GPU")

        # arrange
        any_positive_number = 44.0
        ff = Formulae(
            isotope_relaxation_timescale=variant,
            isotope_diffusivity_ratios="HellmannAndHarvey2020",
            isotope_equilibrium_fractionation_factors="VanHook1968",
        )
        n_sd = 1
        attribute_name = f"Bolin number for {heavy_isotope}"

        builder = Builder(
            n_sd=n_sd,
            backend=backend_class(formulae=ff),
            environment=Box(dt=np.nan, dv=np.nan),
        )
        builder.request_attribute(attribute_name)
        builder.request_attribute(f"delta_{heavy_isotope}")

        for iso in HEAVY_ISOTOPES:
            builder.particulator.environment[f"molality {iso} in dry air"] = 0
        builder.particulator.environment[f"molality {heavy_isotope} in dry air"] = 0.44

        attr = {
            "multiplicity": np.ones(n_sd),
            "signed water mass": np.full(n_sd, si.ng),
            **{
                f"moles_{iso}": np.full(
                    n_sd, moles_heavy if iso == heavy_isotope else 0.0
                )
                for iso in HEAVY_ISOTOPES
            },
        }
        particulator = builder.build(attributes=attr)
        particulator.environment["RH"] = relative_humidity
        particulator.environment["T"] = any_positive_number
        particulator.environment["dry_air_density"] = any_positive_number

        # act
        value = particulator.attributes[attribute_name].data

        # assert
        np.testing.assert_approx_equal(
            np.sign(value), expected_sign_of_tau, significant=5
        )

    @staticmethod
    def test_moles(
        backend_class,
        m_t=1 * si.ng,
    ):
        # arrange
        formulae = Formulae()
        attributes = {
            "multiplicity": np.asarray([0]),
            "signed water mass": np.asarray([m_t]),
        }
        for isotope in HEAVY_ISOTOPES:
            attributes[f"moles_{isotope}"] = np.asarray([44])

        builder = Builder(
            n_sd=1,
            backend=backend_class(formulae=formulae),
            environment=Box(dv=np.nan, dt=-1 * si.s),
        )
        builder.request_attribute("moles light water")
        builder.request_attribute("moles_16O")
        particulator = builder.build(attributes=attributes, products=())

        # assert
        np.testing.assert_approx_equal(
            particulator.attributes["moles light water"].data[0],
            particulator.attributes["moles_16O"].data[0]
            - particulator.attributes["moles_2H"].data[0]
            - particulator.attributes["moles_3H"].data[0],
            significant=5,
        )
