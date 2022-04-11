# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest

from PySDM import Formulae


@pytest.mark.parametrize(
    "constants",
    (
        {"MAC": 1.0, "HAC": 1.0},
        {"MAC": 1.0, "HAC": 0.1},
        {"MAC": 0.1, "HAC": 1.0},
        {"MAC": 0.1, "HAC": 0.1},
    ),
)
def test_accomodation_coefficients(constants):
    # arrange
    formulae = Formulae(constants=constants)
    D = 1
    K = 2
    r = 3
    lmbd = 4

    # act
    D_dk = formulae.diffusion_kinetics.D(D, r, lmbd)
    K_dk = formulae.diffusion_kinetics.K(K, r, lmbd)

    # assert
    Kn = lmbd / r
    xx_D = 4 / 3 / constants["MAC"]
    np.testing.assert_almost_equal(
        D_dk, D * (1 + Kn) / (1 + (xx_D + 0.377) * Kn + xx_D * Kn**2)
    )
    xx_K = 4 / 3 / constants["HAC"]
    np.testing.assert_almost_equal(
        K_dk, K * (1 + Kn) / (1 + (xx_K + 0.377) * Kn + xx_K * Kn**2)
    )
