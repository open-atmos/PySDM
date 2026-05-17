"""
checking consistency with values in the paper for Figure 1 and Figure 2
"""

from pathlib import Path

import numpy as np
import pytest
from open_atmos_jupyter_utils import notebook_vars

from examples.PySDM_examples import Arabas_and_Pawlowska_2011
from examples.PySDM_examples.Long_1974 import settings

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
        settings = variables_fig2["settings"]

        assert settings.dt == 0.25 * si.s
        assert settings.mass_of_dry_air == 1000 * si.kg
        assert settings.p0 == 1000 * si.hPa
        assert settings.RH0 == 0.99 * si.dimensionless
        assert settings.T0 == 280 * si.K
        assert settings.w == 0.25 * si.m / si.s

    @staticmethod
    def test_kappa_values(variables_fig2):
        settings = variables_fig2["settings"]

        np.testing.assert_approx_equal(
            actual=settings.kappa_sea_salt,
            desired=1.28 * si.dimensionless,
            significant=4,
        )

        np.testing.assert_approx_equal(
            actual=settings.kappa_sulphate,
            desired=0.61 * si.dimensionless,
            significant=4,
        )


# class TestFigure1:
