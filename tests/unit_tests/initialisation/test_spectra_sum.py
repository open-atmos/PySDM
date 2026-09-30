# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest

from PySDM.initialisation.spectra import Sum, Gaussian


@pytest.mark.parametrize("fun", ("pdf", "cdf"))
@pytest.mark.parametrize("x", (-10, -1, 0, 1, 10))
def test_spectra_sum(fun, x):
    # arrange
    n1, n2 = 100, 200
    dist1 = Gaussian(norm_factor=n1, loc=1, scale=0.5)
    dist2 = Gaussian(norm_factor=n2, loc=2, scale=1.5)

    # act
    sut = Sum((dist1, dist2))

    # assert
    np.testing.assert_approx_equal(
        actual=getattr(sut, fun)(x),
        desired=(
            getattr(dist1, fun)(x) * n1 / (n1 + n2)
            + getattr(dist2, fun)(x) * n2 / (n1 + n2)
        ),
        significant=15,
    )
