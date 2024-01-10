from pathlib import Path
from PySDM.physics import si
from PySDM_examples import Graf_2019
from PySDM_examples.utils import notebook_vars


def test_fig_4(plot=False):
    # arrange
    vars = notebook_vars(
        file=Path(Graf_2019.__file__).parent / "figure_4.ipynb",
        plot=plot
    )

    # act
    actual = vars["levels"]["CB"] + vars["alt_initial"]
    expected = 1 * si.km

    # assert
    assert abs(expected - actual) < 50 * si.m
