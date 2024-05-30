""" tests ensuring values on plots match those in the paper """

from pathlib import Path

import pytest

from PySDM_examples.utils import notebook_vars
from PySDM_examples import Rozanski_and_Sonntag_1982

PLOT = False


@pytest.fixture(scope="session", name="variables")
def variables_fixture():
    return notebook_vars(
        file=Path(Rozanski_and_Sonntag_1982.__file__).parent / "figs_4_5_6.ipynb",
        plot=PLOT,
    )


def test_fig_5_asymptotes(variables):
    pass
