"""
test checking values in table against paper
"""

from pathlib import Path
from collections import defaultdict
import numpy as np
import pytest
from PySDM_examples.utils import notebook_vars
from PySDM_examples import Bolin_1958


@pytest.fixture(scope="session", name="notebook_variables")
def notebook_variables_fixture():
    return notebook_vars(
        file=Path(Bolin_1958.__file__).parent / "table_1.ipynb",
        plot=False,
    )


class TestTable1:
    COLUMNS = {
        "row": None,
        "radius_cm": "radius [cm]",
        "adjustment_time": "adjustment time [s]",
        "terminal_velocity": "terminal velocity [m/s]",
        "distance": "distance [m]",
    }

    @staticmethod
    @pytest.mark.parametrize(
        ",".join(COLUMNS.keys()),
        (
            (0, 0.005, 3.3, 0.27, 0.9),
            (1, 0.01, 7.1, 0.72, 5.1),
            (2, 0.025, 33, 2.1, 69),
            (3, 0.05, 93, 4.0, 370),
            (4, 0.075, 165, 5.4, 890),
            (5, 0.1, 245, 6.5, 1600),
            (6, 0.15, 365, 8.1, 3000),
            (7, 0.2, 435, 8.8, 3800),
        ),
    )
    @pytest.mark.parametrize(
        "column_var, column_label",
        {k: v for k, v in COLUMNS.items() if k != "row"}.items(),
    )
    def test_against_values_from_the_paper(
        notebook_variables,
        column_var,
        column_label,
        row,
        radius_cm,
        adjustment_time,
        terminal_velocity,
        distance,
    ):
        np.testing.assert_allclose(
            actual=notebook_variables["data"][column_label][row],
            desired=locals()[column_var],
            rtol=defaultdict(lambda: 0.53, radius_cm=0)[column_var],
        )
