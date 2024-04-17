""" tests ensuring values on plots match those in the paper """

from pathlib import Path

import numpy as np
import pytest

from PySDM_examples.utils import notebook_vars
from PySDM_examples import Graf_et_al_2019

from PySDM.physics.constants import PER_MILLE

PLOT = False


@pytest.fixture(scope="session", name="variables")
def variables_fixture():
    return notebook_vars(
        file=Path(Graf_et_al_2019.__file__).parent / "Table_1.ipynb", plot=PLOT
    )


@pytest.mark.parametrize(
    "temp_celsius, phases, case, var, diff",
    (
        (20, "l_v", "A", "diff_delta_2H", 78.2 * PER_MILLE),
        (20, "l_v", "A", "diff_delta_18O", 9.7 * PER_MILLE),
        (20, "l_v", "A", "diff_d_excess", 0.7 * PER_MILLE),
        (20, "l_v", "B", "diff_delta_2H", 68 * PER_MILLE),
        (20, "l_v", "B", "diff_delta_18O", 9.5 * PER_MILLE),
        (20, "l_v", "B", "diff_d_excess", -8.4 * PER_MILLE),
        (0, "l_v", "A", "diff_delta_2H", 103.3 * PER_MILLE),
        (0, "l_v", "A", "diff_delta_18O", 11.6 * PER_MILLE),
        (0, "l_v", "A", "diff_d_excess", 10.5 * PER_MILLE),
        (0, "l_v", "B", "diff_delta_2H", 89.9 * PER_MILLE),
        (0, "l_v", "B", "diff_delta_18O", 11.4 * PER_MILLE),
        (0, "l_v", "B", "diff_d_excess", -1.6 * PER_MILLE),
        (0, "s_v", "A", "diff_delta_2H", 121.3 * PER_MILLE),
        (0, "s_v", "A", "diff_delta_18O", 15.1 * PER_MILLE),
        (0, "s_v", "A", "diff_d_excess", 0.6 * PER_MILLE),
        (0, "s_v", "B", "diff_delta_2H", 105.4 * PER_MILLE),
        (0, "s_v", "B", "diff_delta_18O", 14.9 * PER_MILLE),
        (0, "s_v", "B", "diff_d_excess", -13.4 * PER_MILLE),
    ),
)
# pylint: disable=too-many-arguments
def test_table_1(variables, temp_celsius, phases, case, var, diff):
    # arrange
    three_for_per_mille = 3
    decimal = three_for_per_mille + 1

    # act
    actual = variables["table_data"][temp_celsius][phases][case][var]

    # assert
    np.testing.assert_almost_equal(actual=actual, desired=diff, decimal=decimal)
