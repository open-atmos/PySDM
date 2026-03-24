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
            (0, 12 * si.um, 12 * si.um),
            (1, 16 * si.um, 12 * si.um),
            (3, 78 * si.um, 12 * si.um),
        ),
    )
    def test_continental_mode(notebook_local_variables, it, r_mode_G, r_mode_L):
        np.testing.assert_allclose(
            actual=notebook_local_variables["fig_10_11_data"]["radius_bins_edges"][
                np.argmax(
                    notebook_local_variables["fig_10_11_data"]["G"]["dv/dlnr"][it]
                )
            ],
            desired=r_mode_G,
            rtol=.33,
        )
        np.testing.assert_allclose(
            actual=notebook_local_variables["fig_10_11_data"]["radius_bins_edges"][
                np.argmax(
                    notebook_local_variables["fig_10_11_data"]["L"]["dv/dlnr"][it]
                )
            ],
            desired=r_mode_L,
            rtol=.33,
        )

    @staticmethod
    @pytest.mark.parametrize(
        "it, r_mode_G, r_mode_L",
        (
            (0, 19 * si.um, 19 * si.um),
            (1, 23 * si.um, 19 * si.um),
            (3, 49 * si.um, 17 * si.um),
        ),
    )
    def test_marine_mode(notebook_local_variables, it, r_mode_G, r_mode_L):
        np.testing.assert_allclose(
            actual=notebook_local_variables["fig_13_14_data"]["radius_bins_edges"][
                np.argmax(
                    notebook_local_variables["fig_13_14_data"]["G"]["dv/dlnr"][it]
                )
            ],
            desired=r_mode_G,
            rtol=.33,
        )
        np.testing.assert_allclose(
            actual=notebook_local_variables["fig_13_14_data"]["radius_bins_edges"][
                np.argmax(
                    notebook_local_variables["fig_13_14_data"]["L"]["dv/dlnr"][it]
                )
            ],
            desired=r_mode_L,
            rtol=.33,
        )
