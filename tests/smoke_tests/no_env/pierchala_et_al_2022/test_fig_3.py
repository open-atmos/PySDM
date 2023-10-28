"""
regression tests checking values from the plots
"""
from pathlib import Path

import nbformat
import numpy as np
import pytest
from PySDM_examples import Pierchala_et_al_2022

from PySDM.physics.constants_defaults import PER_MEG, PER_MILLE

PLOT = False


@pytest.fixture(scope="session", name="notebook_local_variables")
def notebook_local_variables_fixture():
    notebook = nbformat.read(
        Path(Pierchala_et_al_2022.__file__).parent / "fig_3.ipynb", nbformat.NO_CONVERT
    )
    context = {}
    for cell in notebook.cells:
        if cell.cell_type != "markdown":
            lines = cell.source.splitlines()
            for i, line in enumerate(lines):
                if line.strip().startswith("!"):
                    lines[i] = line.replace("!", "pass #")
                if line.strip().startswith("show_plot("):
                    lines[i] = line.replace(
                        "show_plot() #",
                        "pyplot.show(" if PLOT else "pyplot.gca().clear() #",
                    )

            exec("\n".join(lines), context)  # pylint: disable=exec-used
    return context


class TestFig3:
    @staticmethod
    @pytest.mark.parametrize(
        "isotope, F, enrichment",
        (
            ("18O", 1.0, 0),
            ("18O", 0.3, 25.5 * PER_MILLE),
            ("17O", 1.0, 0),
            ("17O", 0.3, 13.3 * PER_MILLE),
            ("2H", 1.0, 0),
            ("2H", 0.3, 109 * PER_MILLE),
        ),
    )
    def test_top_left_panel(notebook_local_variables, isotope, F, enrichment):
        index = np.where(notebook_local_variables["F"] == F)
        np.testing.assert_approx_equal(
            actual=notebook_local_variables["enrichments"][isotope][index],
            desired=enrichment,
            significant=3,
        )

    @staticmethod
    @pytest.mark.parametrize(
        "label, F, excess",
        (
            ("d-excess", 1.0, 7.68 * PER_MILLE),
            ("d-excess", 0.4, -68.4 * PER_MILLE),
            ("17O-excess", 1.0, 29.04 * PER_MEG),
            ("17O-excess", 0.3, -70.2 * PER_MEG),
        ),
    )
    def test_bottom_left_panel(notebook_local_variables, label, F, excess):
        index = np.where(notebook_local_variables["F"] == F)
        excesses = notebook_local_variables["excess"]
        deltas = notebook_local_variables["deltas"]
        np.testing.assert_approx_equal(
            actual={
                "d-excess": excesses.excess_d(deltas["2H"], deltas["18O"]),
                "17O-excess": excesses.excess_17O(deltas["17O"], deltas["18O"]),
            }[label][index],
            desired=excess,
            significant=3,
        )

    @staticmethod
    @pytest.mark.parametrize(
        "delta_18O, delta_2H",
        ((-8.71 * PER_MILLE, -62 * PER_MILLE), (16.5 * PER_MILLE, 40.5 * PER_MILLE)),
    )
    def test_top_right_panel(notebook_local_variables, delta_18O, delta_2H):
        eps = 0.01 * PER_MILLE
        index = np.where(
            abs(notebook_local_variables["deltas"]["18O"] - delta_18O) < eps
        )
        np.testing.assert_approx_equal(
            actual=notebook_local_variables["deltas"]["2H"][index],
            desired=delta_2H,
            significant=3,
        )

    @staticmethod
    @pytest.mark.parametrize(
        "delta_18O, delta_2H",
        ((-8.71 * PER_MILLE, -60.3 * PER_MILLE), (6 * PER_MILLE, 57.6 * PER_MILLE)),
    )
    def test_gmvl(notebook_local_variables, delta_18O, delta_2H):
        cd = notebook_local_variables["const"]
        eps = 0.1 * PER_MILLE
        x = np.linspace(-10 * PER_MILLE, 10 * PER_MILLE, 100)
        y = x * cd.CRAIG_1961_SLOPE_COEFF + cd.CRAIG_1961_INTERCEPT_COEFF
        index = np.where(abs(x - delta_18O) < eps)
        np.testing.assert_approx_equal(actual=y[index], desired=delta_2H, significant=3)

    @staticmethod
    @pytest.mark.parametrize(
        "label, delta_18O, excess",
        (
            ("d-excess", -8.5 * PER_MILLE, 6.46 * PER_MILLE),
            ("d-excess", 16.5 * PER_MILLE, -91.6 * PER_MILLE),
            ("17O-excess", -8.5 * PER_MILLE, 27.9 * PER_MEG),
            ("17O-excess", 16.5 * PER_MILLE, -70.2 * PER_MEG),
        ),
    )
    def test_bottom_right_panel(notebook_local_variables, label, delta_18O, excess):
        eps = 0.1 * PER_MILLE
        deltas = notebook_local_variables["deltas"]
        index = np.where(abs(deltas["18O"] - delta_18O) < eps)
        excesses = notebook_local_variables["excess"]
        np.testing.assert_approx_equal(
            actual={
                "d-excess": excesses.excess_d(deltas["2H"], deltas["18O"]),
                "17O-excess": excesses.excess_17O(deltas["17O"], deltas["18O"]),
            }[label][index],
            desired=excess,
            significant=3,
        )
