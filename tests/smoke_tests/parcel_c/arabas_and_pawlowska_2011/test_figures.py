"""
checking consistency with values in the paper for Figure 1 and Figure 2
"""

from pathlib import Path

import numpy as np
import pytest
from open_atmos_jupyter_utils import notebook_vars

from PySDM_examples import Arabas_and_Pawlowska_2011

from PySDM.physics import si

PlOT = False


@pytest.fixture(scope="session", name="variables_fig1")
def variables_fig1_fixture():
    return notebook_vars(
        file=Path(Arabas_and_Pawlowska_2011.__file__).parent / "fig_1.ipynb", plot=PlOT
    )


@pytest.fixture(scope="session", name="variables_fig2")
def variables_fig2_fixture():
    return notebook_vars(
        file=Path(Arabas_and_Pawlowska_2011.__file__).parent / "fig_2.ipynb", plot=PlOT
    )


class TestInitial:
    @staticmethod
    def test_initial_conditions(variables_fig2):
        settings_ = variables_fig2["settings"]

        assert settings_.dt == 0.25 * si.s
        assert settings_.mass_of_dry_air == 1000 * si.kg
        assert settings_.p0 == 1000 * si.hPa
        assert settings_.RH0 == 0.99 * si.dimensionless
        assert settings_.T0 == 280 * si.K
        assert settings_.w == 0.25 * si.m / si.s

    @staticmethod
    def test_kappa_values(variables_fig2):
        settings_ = variables_fig2["settings"]

        np.testing.assert_approx_equal(
            actual=settings_.kappa_sea_salt,
            desired=1.28 * si.dimensionless,
            significant=4,
        )

        np.testing.assert_approx_equal(
            actual=settings_.kappa_sulphate,
            desired=0.61 * si.dimensionless,
            significant=4,
        )


class TestFigure1:
    @staticmethod
    @pytest.mark.parametrize(
        "variable", ("r_dry_ss", "r_dry_su", "r_wet0_ss", "r_wet0_su")
    )
    def test_panel_a_x(variables_fig1, variable):
        assert (
            1e-3 * si.um
            <= min(variables_fig1[variable])
            < max(variables_fig1[variable])
            <= 1e2 * si.um
        )

    @staticmethod
    @pytest.mark.parametrize(
        "variable", ("y_dry_ss", "y_dry_su", "y_wet0_ss", "y_wet0_su")
    )
    def test_panel_a_y(variables_fig1, variable):
        assert (
            1e-3 / (si.mg * si.um)
            <= max(variables_fig1[variable])
            <= 1e7 / (si.mg * si.um)
        )

    @staticmethod
    def test_integral(variables_fig1):
        total = np.dot(
            variables_fig1["y_dry_su"][:-1], np.diff(variables_fig1["r_dry_su"])
        )
        assert 80 / si.mg < total < 120 / si.mg


class TestFigure2:
    @staticmethod
    def test_radius_range(variables_fig2):
        output = variables_fig2["output"]
        assert np.all(output["radius"] / si.um > 0)
        assert 1e-3 <= np.max(output["radius"] / si.um) <= 1e2

    @staticmethod
    def test_relative_humidity_range(variables_fig2):
        output = variables_fig2["output"]
        rh_percent = np.asarray(output["RH"]) * 100
        assert 100 < np.max(rh_percent) < 100.5
