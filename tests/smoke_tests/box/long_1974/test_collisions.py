# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
from pathlib import Path
import numpy as np
import pytest
from open_atmos_jupyter_utils import notebook_vars
from PySDM_examples import Long_1974

from PySDM.physics import si

PLOT = False


@pytest.fixture(scope="session", name="notebook_local_variables")
def notebook_local_variables_fixture():
    return notebook_vars(
        Path(Long_1974.__file__).parent / "fig10_11_13_14.ipynb", plot=PLOT
    )


class TestLongFigs:
    @staticmethod
    @pytest.mark.parametrize(
        "it, r_mode_G, r_mode_L",
        (
            (0, 12.0 * si.um, 12.0 * si.um),
            (1, 19.2 * si.um, 12.0 * si.um),
            (3, 69.7 * si.um, 11.3 * si.um),
        ),
    )
    def test_continental_mode(notebook_local_variables, it, r_mode_G, r_mode_L):
        np.testing.assert_approx_equal(
            actual=notebook_local_variables["fig_10_11_data"]["radius_bins_edges"][
                np.argmax(
                    notebook_local_variables["fig_10_11_data"]["G"]["dv/dlnr"][it]
                )
            ],
            desired=r_mode_G,
            significant=2,
        )
        np.testing.assert_approx_equal(
            actual=notebook_local_variables["fig_10_11_data"]["radius_bins_edges"][
                np.argmax(
                    notebook_local_variables["fig_10_11_data"]["L"]["dv/dlnr"][it]
                )
            ],
            desired=r_mode_L,
            significant=2,
        )

    @staticmethod
    @pytest.mark.parametrize(
        "it, r_mode_G, r_mode_L",
        (
            (0, 18.7 * si.um, 18.7 * si.um),
            (1, 23.1 * si.um, 18.7 * si.um),
            (3, 43.2 * si.um, 18.7 * si.um),
        ),
    )
    def test_marine_mode(notebook_local_variables, it, r_mode_G, r_mode_L):
        np.testing.assert_approx_equal(
            actual=notebook_local_variables["fig_13_14_data"]["radius_bins_edges"][
                np.argmax(
                    notebook_local_variables["fig_13_14_data"]["G"]["dv/dlnr"][it]
                )
            ],
            desired=r_mode_G,
            significant=2,
        )
        np.testing.assert_approx_equal(
            actual=notebook_local_variables["fig_13_14_data"]["radius_bins_edges"][
                np.argmax(
                    notebook_local_variables["fig_13_14_data"]["L"]["dv/dlnr"][it]
                )
            ],
            desired=r_mode_L,
            significant=2,
        )
