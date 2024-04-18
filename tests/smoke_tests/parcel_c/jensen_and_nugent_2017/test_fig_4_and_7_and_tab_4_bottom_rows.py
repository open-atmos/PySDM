# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
from pathlib import Path
import numpy as np
import pytest
from scipy import signal
from PySDM_examples.utils import notebook_vars
from PySDM_examples import Jensen_and_Nugent_2017
from PySDM.physics.constants import PER_CENT
from PySDM.physics import si
from .test_fig_3_and_tab_4_upper_rows import find_cloud_base_index, find_max_alt_index

PLOT = False
N_SD = Jensen_and_Nugent_2017.simulation.N_SD_NON_GCCN + np.count_nonzero(
    Jensen_and_Nugent_2017.table_3.NA
)


@pytest.fixture(scope="session", name="variables")
def variables_fixture():
    return notebook_vars(
        file=Path(Jensen_and_Nugent_2017.__file__).parent
        / "Fig_4_and_7_and_Tab_4_bottom_rows.ipynb",
        plot=PLOT,
    )


class TestFig4And7:
    @staticmethod
    def test_height_range(variables):
        """note: in the plot the y-axis has cloud-base height subtracted, here not"""
        z_minus_z0 = (
            np.asarray(variables["output"]["products"]["z"]) - variables["settings"].z0
        )
        epsilon = 1 * si.m
        assert 0 <= min(z_minus_z0) < max(z_minus_z0) < 600 * si.m + epsilon

    @staticmethod
    def test_cloud_base_height(variables):
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
        assert 0.4 * PER_CENT < np.nanmax(supersaturation) < 0.5 * PER_CENT

    class TestFig4:

        @staticmethod
        @pytest.mark.parametrize(
            "drop_id, activated, grow_on_descent",
            (
                [(drop_id, False, False) for drop_id in range(0, int(0.15 * N_SD))]
                + [
                    (drop_id, True, False)
                    for drop_id in range(int(0.25 * N_SD), int(0.6 * N_SD))
                ]
                + [(drop_id, True, True) for drop_id in range(int(0.777 * N_SD), N_SD)]
            ),
        )
        def test_grow_vs_evaporation_on_descent(
            variables, drop_id, activated, grow_on_descent
        ):
            # arrange
            cb_idx = find_cloud_base_index(variables["output"]["products"])
            ma_idx = find_max_alt_index(variables["output"]["products"])
            radii = variables["output"]["attributes"]["radius"][drop_id]
            r1 = radii[0]
            r2 = radii[cb_idx]
            r3 = radii[ma_idx]
            r4 = radii[-1]

            activated_actual = r1 < r2 < r3
            assert activated == activated_actual

            if grow_on_descent:
                assert r3 < r4
            else:
                assert r3 > r4

        @staticmethod
        def test_maximal_size_of_largest_droplet(variables):
            np.testing.assert_approx_equal(
                max(variables["output"]["attributes"]["radius"][-1]),
                57 * si.um,
                significant=2,
            )

        @staticmethod
        def test_initial_size_of_largest_droplet(variables):
            np.testing.assert_approx_equal(
                min(variables["output"]["attributes"]["radius"][-1]),
                19 * si.um,
                significant=2,
            )

    class TestFig7:  # pylint: disable=too-few-public-methods
        @staticmethod
        @pytest.mark.parametrize(
            "mask_label, var, dry_radius_um, value_range",
            (
                ("ascent", "SS_eq", 0.1, (0, 0.1 * PER_CENT)),
                ("ascent", "SS_eq", 2, (-0.75 * PER_CENT, 0)),
                ("ascent", "SS_eq", 9, (-2.01 * PER_CENT, -0.6 * PER_CENT)),
                ("descent", "SS_eq", 0.1, (0, 0.1 * PER_CENT)),
                ("descent", "SS_eq", 2, (-0.75 * PER_CENT, 0)),
                ("descent", "SS_eq", 9, (-1 * PER_CENT, -0.25 * PER_CENT)),
                ("ascent", "SS_ef", 0.1, (0, 0.5 * PER_CENT)),
                ("ascent", "SS_ef", 2, (0, 1 * PER_CENT)),
                ("ascent", "SS_ef", 9, (0.75 * PER_CENT, 2.5 * PER_CENT)),
                ("descent", "SS_ef", 0.1, (-0.2 * PER_CENT, 0.1 * PER_CENT)),
                ("descent", "SS_ef", 2, (-0.5 * PER_CENT, 0.25 * PER_CENT)),
                ("descent", "SS_ef", 9, (0.3 * PER_CENT, 0.8 * PER_CENT)),
            ),
        )
        def test_equilibrium_supersaturation(
            variables, *, mask_label, var, dry_radius_um, value_range
        ):
            mask = np.logical_and(
                variables["masks"][mask_label], variables["height_above_cloud_base"] > 0
            )
            assert (variables[var][dry_radius_um][mask] > value_range[0]).all()
            assert (variables[var][dry_radius_um][mask] < value_range[1]).all()

    class TestTable4bottom:  # pylint: disable=too-few-public-methods

        @staticmethod
        @pytest.mark.parametrize(
            "mask_label, height, mr_sw_rd",
            (
                ("ascent", 50, (5.26, 0.45, 0.093)),
                ("ascent", 100, (6.73, 0.45, 0.066)),
                ("ascent", 150, (7.73, 0.43, 0.055)),
                ("ascent", 200, (8.52, 0.41, 0.048)),
                ("ascent", 250, (9.19, 0.41, 0.044)),
                ("ascent", 300, (9.77, 0.40, 0.041)),
                ("descent", 300, (9.77, 0.40, 0.041)),
                ("descent", 250, (9.21, 0.43, 0.047)),
                ("descent", 200, (8.56, 0.46, 0.054)),
                ("descent", 150, (7.78, 0.51, 0.065)),
                ("descent", 100, (6.79, 0.57, 0.084)),
                ("descent", 50, (5.38, 0.69, 0.129)),
            ),
        )
        def test_table_4_bottom_rows(
            variables,
            mask_label,
            height,
            mr_sw_rd,
        ):
            # arrange
            tolerance = 0.2
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
                        actual=float(row[2]),
                        desired=spectral_width,
                        rtol=tolerance,
                    )
                    np.testing.assert_allclose(
                        actual=float(row[3]),
                        desired=relative_dispersion,
                        rtol=tolerance,
                    )
