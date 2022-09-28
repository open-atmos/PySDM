# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
from PySDM import formulae


def test_c_inline():
    # arrange
    def fun(_, xxx):
        return min(
            xxx,
            2,
        )

    # act
    c_code = formulae._c_inline(
        fun, constants={"pi": 3.14}, xxx=0
    )  # pylint: disable=protected-access

    # assert
    assert ", )" not in c_code
