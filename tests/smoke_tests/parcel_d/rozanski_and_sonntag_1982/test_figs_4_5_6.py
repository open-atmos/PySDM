""" tests ensuring values on plots match those in the paper """

from pathlib import Path

import numpy as np
import pytest

from PySDM_examples.utils import notebook_vars
from PySDM_examples import Rozanski_and_Sonntag_1982
from PySDM.physics import in_unit

PLOT = False


@pytest.fixture(scope="session", name="variables")
def variables_fixture():
    return notebook_vars(
        file=Path(Rozanski_and_Sonntag_1982.__file__).parent / "figs_4_5_6.ipynb",
        plot=PLOT,
    )


class TestFigs456:
    @staticmethod
    def test_fig_5_vapour_asymptote(variables):
        delta_vapour_at_the_cloud_top_per_mille = in_unit(
            variables["formulae"].trivia.isotopic_ratio_2_delta(
                ratio=variables["data"][variables["PLOT_KEY"]]["CT"]["Rv_2H"][1:],
                reference_ratio=variables["const"].VSMOW_R_2H,
            ),
            variables["const"].PER_MILLE,
        )

        d_delta = np.diff(delta_vapour_at_the_cloud_top_per_mille)
        dd_delta = np.diff(d_delta)

        mean = np.mean(delta_vapour_at_the_cloud_top_per_mille)
        assert (d_delta < 0).all()
        assert (dd_delta > 0).all()
        assert (np.abs(d_delta / mean)[-10:] < 0.02).all()
        np.testing.assert_approx_equal(
            delta_vapour_at_the_cloud_top_per_mille[-1], -460, significant=2
        )

    @staticmethod
    def test_fig_5_rain_at_the_cloud_base(variables):
        delta_rain_at_the_cloud_base_per_mille = in_unit(
            variables["formulae"].trivia.isotopic_ratio_2_delta(
                ratio=variables["data"][variables["PLOT_KEY"]]["CB"]["Rr_2H"][1:],
                reference_ratio=variables["const"].VSMOW_R_2H,
            ),
            variables["const"].PER_MILLE,
        )

        d_delta = np.diff(delta_rain_at_the_cloud_base_per_mille)
        dd_delta = np.diff(d_delta)

        mean = np.mean(delta_rain_at_the_cloud_base_per_mille)
        assert (d_delta[len(d_delta) // 4 :] < 0).all()
        assert (dd_delta[len(dd_delta) // 4 :] > 0).all()
        assert (np.abs(d_delta / mean)[-10:] < 0.02).all()
        np.testing.assert_approx_equal(
            delta_rain_at_the_cloud_base_per_mille[-1], -16, significant=2
        )
