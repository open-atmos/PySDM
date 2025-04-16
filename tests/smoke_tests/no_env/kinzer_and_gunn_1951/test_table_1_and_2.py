"""
test if values in table1 (table2) are increasing (decreasing) in each column
"""

import numpy as np
import pytest

from PySDM_examples.Kinzer_And_Gunn_1951.table_1_and_2 import table1, table2


@pytest.mark.parametrize("temperature", (0, 20, 30, 40))
@pytest.mark.parametrize("table, slope_sign", ((table1, 1), (table2, -1)))
def test_table_1_monotonicity(table, temperature, slope_sign):
    # Arrange
    values = np.array([x for x in table[f"{temperature} [deg C]"] if x != 0])

    # Act
    differences = values[1:] - values[:-1]
    signs = np.sign(differences)

    # Assert
    np.testing.assert_equal(actual=signs, desired=slope_sign)
