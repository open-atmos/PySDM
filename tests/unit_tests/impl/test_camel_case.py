"""
test checking CamelCase conversion routine
"""

from typing import Tuple
import pytest
from PySDM.impl.camel_case import camel_case_to_words


@pytest.mark.parametrize(
    "in_out_pair", (("CPUTime", "CPU time"), ("WallTime", "wall time"))
)
def test_camel_case_to_words(in_out_pair: Tuple[str, str]):
    # arrange
    test_input, expected_output = in_out_pair

    # act
    actual_output = camel_case_to_words(test_input)

    # assert
    assert actual_output == expected_output
