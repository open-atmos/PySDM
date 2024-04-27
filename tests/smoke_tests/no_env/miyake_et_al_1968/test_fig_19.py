"""
regression tests checking values from the plots in Fig 19
"""

from pathlib import Path

import numpy as np
import pytest

from PySDM_examples.utils.notebook_vars import notebook_vars
from PySDM_examples import Miyake_et_al_1968

PLOT = False


@pytest.fixture(scope="session", name="notebook_local_variables")
def notebook_local_variables_fixture():
    return notebook_vars(
        Path(Miyake_et_al_1968.__file__).parent / "fig_19.ipynb", plot=PLOT
    )


class TestFig19:
    @staticmethod
    def test_values(notebook_local_variables):
        plot_x = notebook_local_variables["plot_x"]
        assert 0.6 < min(plot_x) < 0.8
        assert 1.9 < max(plot_x) < 2.1

        for plot_y in notebook_local_variables["plot_y"].values():
            assert 0 < min(plot_y) < 5
            assert 1.25 < max(plot_y) < 20

    @staticmethod
    def test_temperature_dependence(notebook_local_variables):
        for variant in notebook_local_variables["VENTILATION_VARIANTS"]:
            for iso in ("18O", "17O", "2H"):
                assert (
                    notebook_local_variables["plot_y"][f"{variant}-{283.15}-{iso}"]
                    < notebook_local_variables["plot_y"][f"{variant}-{293.15}-{iso}"]
                ).all()

    @staticmethod
    def test_isotope_dependence(notebook_local_variables):
        for temp in (283.15, 293.15):
            for variant in notebook_local_variables["VENTILATION_VARIANTS"]:
                assert (
                    notebook_local_variables["plot_y"][f"{variant}-{temp}-18O"]
                    > notebook_local_variables["plot_y"][f"{variant}-{temp}-17O"]
                ).all()
                assert (
                    notebook_local_variables["plot_y"][f"{variant}-{temp}-17O"]
                    > notebook_local_variables["plot_y"][f"{variant}-{temp}-2H"]
                ).all()

    @staticmethod
    def test_monotonic(notebook_local_variables):
        assert (np.diff(notebook_local_variables["plot_x"]) > 0).all()
        for key in notebook_local_variables["plot_y"]:
            assert (np.diff(notebook_local_variables["plot_y"][key]) < 0).all()
