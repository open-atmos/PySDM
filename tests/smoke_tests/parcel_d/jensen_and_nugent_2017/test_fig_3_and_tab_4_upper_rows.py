# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring

from pathlib import Path

import numpy as np
import pytest
from scipy import signal

from PySDM_examples.utils import notebook_vars
from PySDM_examples import Jensen_and_Nugent_2017
from PySDM.physics.constants import PER_CENT

from PySDM.physics import si

PLOT = False


def find_cloud_base_index(products):
    for index, value in enumerate(products["S_max"]):
        if value > 0:
            cloud_base_index = index
            break
    return cloud_base_index


def find_max_alt_index(products):
    return np.argmax(products["z"])


@pytest.fixture(scope="session", name="variables")
def variables_fixture():
    return notebook_vars(
        file=Path(Jensen_and_Nugent_2017.__file__).parent
        / "Fig_3_and_Tab_4_upper_rows.ipynb",
        plot=PLOT,
    )


class TestFig3:
    @staticmethod
    def test_height_range(variables):
        """note: in the plot the y axis has cloud-base height subtracted, here not"""
        z_minus_z0 = (
            np.asarray(variables["output"]["products"]["z"]) - variables["settings"].z0
        )
        epsilon = 1 * si.m
        assert 0 <= min(z_minus_z0) < max(z_minus_z0) < 600 * si.m + epsilon

    @staticmethod
    def test_cloud_base_height(variables):
        """|------------------> integration
           0         z0           CB
        ---|----------|------------|-------------> z
                      ..............**********
                      subsaturation  supersaturation
        note: in the paper, the CB is defined as altitude where
              maximal supersaturation is attained
        """
        cloud_base_index = find_cloud_base_index(variables["output"]["products"])

        z0 = variables["settings"].z0
        assert (
            290 * si.m
            < variables["output"]["products"]["z"][cloud_base_index] - z0
            < 300 * si.m
        )

    @staticmethod
    def test_supersaturation_maximum(variables):
        supersaturation = np.asarray(variables["output"]["products"]["S_max"])
        assert signal.argrelextrema(supersaturation, np.greater)[0].shape[0] == 1
        assert 0.35 * PER_CENT < np.nanmax(supersaturation) < 0.5 * PER_CENT

    @staticmethod
    @pytest.mark.parametrize(
        "drop_id",
        range(
            Jensen_and_Nugent_2017.simulation.N_SD_NON_GCCN // 4,
            Jensen_and_Nugent_2017.simulation.N_SD_NON_GCCN,
        ),
    )
    def test_radii(variables, drop_id):
        """checks that 75% of the largest aerosol activate and shrink upon descent"""
        # arrange
        cb_idx = find_cloud_base_index(variables["output"]["products"])
        ma_idx = find_max_alt_index(variables["output"]["products"])

        radii = variables["output"]["attributes"]["radius"][drop_id]
        r1 = radii[0]
        r2 = radii[cb_idx]
        r3 = radii[ma_idx]
        r4 = radii[-1]

        assert r1 < r2 < r3
        assert r3 > r4

    @staticmethod
    def test_maximal_size_of_largest_droplet(variables):
        np.testing.assert_approx_equal(
            max(variables["output"]["attributes"]["radius"][-1]),
            12 * si.um,
            significant=2,
        )

    @staticmethod
    @pytest.mark.parametrize(
        "mask_label, height, mr_sw_rd",
        (
            ("ascent", 50, (5.28, 0.44, 0.083)),
            ("ascent", 100, (6.75, 0.38, 0.057)),
            ("ascent", 150, (7.75, 0.36, 0.046)),
            ("ascent", 200, (8.54, 0.34, 0.039)),
            ("ascent", 250, (9.20, 0.32, 0.035)),
            ("ascent", 300, (9.77, 0.31, 0.032)),
            ("descent", 300, (9.77, 0.31, 0.032)),
            ("descent", 250, (9.23, 0.33, 0.036)),
            ("descent", 200, (8.54, 0.36, 0.042)),
            ("descent", 150, (7.80, 0.39, 0.051)),
            ("descent", 100, (6.82, 0.45, 0.066)),
            ("descent", 50, (5.43, 0.57, 0.105)),
        ),
    )
    def test_table_4_upper_rows(variables, mask_label, height, mr_sw_rd):
        # arrange
        tolerance = 0.075
        mean_radius, spectral_width, relative_dispersion = mr_sw_rd

        # act
        actual = variables["table_values"]

        # assert
        for row in actual[mask_label]:
            if int(row[0]) == height:
                np.testing.assert_allclose(
                    actual=float(row[1]), desired=mean_radius, rtol=tolerance
                )
                np.testing.assert_allclose(
                    actual=float(row[2]), desired=spectral_width, rtol=tolerance
                )
                np.testing.assert_allclose(
                    actual=float(row[3]), desired=relative_dispersion, rtol=tolerance
                )
