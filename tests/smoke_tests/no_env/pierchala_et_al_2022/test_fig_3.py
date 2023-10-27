"""
regression tests checking values from the plots
"""
from pathlib import Path

import nbformat
import numpy as np
import pytest
from PySDM_examples import Pierchala_et_al_2022

from PySDM.physics.constants_defaults import PER_MILLE

PLOT = False


@pytest.fixture(scope="session", name="notebook_local_variables")
def notebook_local_variables_fixture():
    notebook = nbformat.read(
        Path(Pierchala_et_al_2022.__file__).parent / "fig_3.ipynb", nbformat.NO_CONVERT
    )

    # act
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
    def test_bottom_left_panel():
        pass  # TODO #1063

    @staticmethod
    def test_top_right_panel():
        pass  # TODO #1063

    @staticmethod
    def test_bottom_right_panel():
        pass  # TODO #1063
