"""
unit tests for isotope-related attributes
"""

import numpy as np
import pytest

from PySDM import Builder
from PySDM.dynamics.isotopic_fractionation import HEAVY_ISOTOPES
from PySDM.environments import Box
from PySDM.physics import constants_defaults, si
from PySDM.physics.trivia import Trivia


def dummy_attrs(length):
    return {
        "water mass": np.asarray([0.666 * si.g] * length),
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
                * attributes["water mass"]
                / heavier_water_molar_mass
            ]
        )
        particulator = builder.build(attributes=attributes)

        # act
        delta = particulator.attributes[f"delta_{isotope}"].to_ndarray()

        # assert
        n_heavy_isotope = attributes[f"moles_{isotope}"][0]
        n_light_water = (
            (1 - heavier_water_specific_content)
            * attributes["water mass"]
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
