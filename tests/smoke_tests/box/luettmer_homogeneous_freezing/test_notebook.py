from pathlib import Path

import pytest

from open_atmos_jupyter_utils import notebook_vars
from PySDM_examples import Luettmer_homogeneous_freezing

PLOT = False  # this will be needed and useful if open_atmos_jupyter_utils.show_plot() is used in the notebook


@pytest.fixture(scope="session", name="variables")
def variables_fixture():
    return notebook_vars(
        file=Path(Luettmer_homogeneous_freezing.__file__).parent
        / "hom_freezing_cloud_droplets.ipynb",
        plot=PLOT,
    )


class TestNotebook:
    @staticmethod
    def test_1(variables):
        assert variables["n_sd"] == 100

    @staticmethod
    def test_2(variables):
        print(len(variables["simulations"]))
