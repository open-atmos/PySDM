# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import pytest

from PySDM.backends.impl_numba.methods.index_methods import draw_random_int


@pytest.mark.parametrize(
    "a, b, u01, expected",
    (
        (0, 100, 0.0, 0),
        (0, 100, 1.0, 100),
        (0, 1, 0.5, 1),
        (0, 1, 0.49, 0),
        (0, 3, 0.49, 1),
        (0, 3, 0.245, 0),
        (0, 2, 0.332, 0),
        (0, 2, 0.333, 0),
        (0, 2, 0.334, 1),
        (0, 2, 0.665, 1),
        (0, 2, 0.666, 1),
        (0, 2, 0.667, 2),
        (0, 2, 0.999, 2),
    ),
)
def test_draw_random_int(a, b, u01, expected):
    # act
    actual = draw_random_int(a, b, u01)

    # assert
    assert actual == expected
