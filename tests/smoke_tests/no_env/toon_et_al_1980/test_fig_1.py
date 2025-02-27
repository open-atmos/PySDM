"""
regression tests checking plotted values
"""

from pathlib import Path

import numpy as np
import pytest

from open_atmos_jupyter_utils import notebook_vars
from PySDM_examples import Toon_et_al_1980
from PySDM.physics import si


PLOT = False


@pytest.fixture(scope="session", name="notebook_variables")
def notebook_variables_fixture():
    return notebook_vars(
        file=Path(Toon_et_al_1980.__file__).parent / "fig_1.ipynb",
        plot=PLOT,
    )


@pytest.mark.parametrize(
    "line_label, expected_match",
    (
        ("CH$_4$ (T=100 K)", {"p": 0.88e-2 * si.mbar, "z": 400 * si.km}),
        ("CH$_4$ (T=100 K)", {"p": 3e0 * si.mbar, "z": 100 * si.km}),
        ("CH$_4$+N$_2$ (T=100 K)", {"p": 2.3e-4 * si.mbar, "z": 400 * si.km}),
        ("CH$_4$+N$_2$ (T=100 K)", {"p": 6e0 * si.mbar, "z": 100 * si.km}),
        ("CH$_4$ partial pressure (T=100 K)", {"p": 7e-4 * si.mbar, "z": 300 * si.km}),
        ("CH$_4$ partial pressure (T=100 K)", {"p": 6e-1 * si.mbar, "z": 100 * si.km}),
        ("CH$_4$ (T=160 K)", {"p": 1.7e-1 * si.mbar, "z": 400 * si.km}),
        ("CH$_4$ (T=160 K)", {"p": 6.6e0 * si.mbar, "z": 100 * si.km}),
        ("CH$_4$+N$_2$ (T=160 K)", {"p": 4.2e-2 * si.mbar, "z": 400 * si.km}),
        ("CH$_4$+N$_2$ (T=160 K)", {"p": 2.4e1 * si.mbar, "z": 100 * si.km}),
        (
            "CH$_4$ partial pressure (T=160 K)",
            {"p": 4.2e-3 * si.mbar, "z": 400 * si.km},
        ),
        ("CH$_4$ partial pressure (T=160 K)", {"p": 2.4e0 * si.mbar, "z": 100 * si.km}),
    ),
)
def test_fig_1_against_values_from_the_paper_plot(
    notebook_variables, line_label: str, expected_match: dict
):
    # arrange
    z = notebook_variables["plot_z"]
    p = notebook_variables["plot_ps"][line_label]

    # act
    idx = np.argmin(np.abs(z - expected_match["z"]))

    # assert
    np.testing.assert_approx_equal(
        actual=p[idx], desired=expected_match["p"], significant=2
    )
