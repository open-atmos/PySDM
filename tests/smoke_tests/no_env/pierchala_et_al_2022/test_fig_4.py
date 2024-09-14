"""
regression tests checking values from the plots in Fig 4
"""

from pathlib import Path

import numpy as np
import pytest

from PySDM_examples.utils.notebook_vars import notebook_vars
from PySDM_examples import Pierchala_et_al_2022

from PySDM.physics.constants import PER_MEG, PER_MILLE

PLOT = False


@pytest.fixture(scope="session", name="notebook_local_variables")
def notebook_local_variables_fixture():
    return notebook_vars(
        Path(Pierchala_et_al_2022.__file__).parent / "fig_4.ipynb", plot=PLOT
    )


class TestFig4:
    @staticmethod
    @pytest.mark.parametrize(
        "RH, delta_18O, delta_2H",
        (
            (0.404, -9.2 * PER_MILLE, -62.0 * PER_MILLE),
            (0.404, 22 * PER_MILLE, 45.8 * PER_MILLE),
            (0.582, -9.2 * PER_MILLE, -62.0 * PER_MILLE),
            (0.582, 16.5 * PER_MILLE, 41.4 * PER_MILLE),
            (0.792, -9.2 * PER_MILLE, -62.0 * PER_MILLE),
            (0.792, 9.5 * PER_MILLE, 36.7 * PER_MILLE),
        ),
    )
    def test_top_panel(notebook_local_variables, RH, delta_18O, delta_2H):
        deltas_per_rh = notebook_local_variables["deltas_per_rh"]
        eps = 0.5 * PER_MILLE
        index = np.where(abs(deltas_per_rh[RH]["18O"] - delta_18O) < eps)
        np.testing.assert_approx_equal(
            actual=delta_2H, desired=deltas_per_rh[RH]["2H"][index], significant=3
        )

    @staticmethod
    @pytest.mark.parametrize(
        "RH, delta_18O, excess_17O",
        (
            (0.404, -9.2 * PER_MILLE, 29.0 * PER_MEG),
            (0.404, 22 * PER_MILLE, -102 * PER_MEG),
            (0.582, -9.2 * PER_MILLE, 29.0 * PER_MEG),
            (0.582, 16.5 * PER_MILLE, -71.7 * PER_MEG),
            (0.792, -9.2 * PER_MILLE, 29.0 * PER_MEG),
            (0.792, 9.5 * PER_MILLE, -17.7 * PER_MEG),
        ),
    )
    def test_bottom_panel(notebook_local_variables, RH, delta_18O, excess_17O):
        deltas_per_rh = notebook_local_variables["deltas_per_rh"]
        eps = 0.5 * PER_MILLE
        index = np.where(abs(deltas_per_rh[RH]["18O"] - delta_18O) < eps)
        np.testing.assert_approx_equal(
            actual=notebook_local_variables[
                "formulae"
            ].isotope_meteoric_water_line_excess.excess_17O(
                deltas_per_rh[RH]["17O"][index], deltas_per_rh[RH]["18O"][index]
            ),
            desired=excess_17O,
            significant=3,
        )
