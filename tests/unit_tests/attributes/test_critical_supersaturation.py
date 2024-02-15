# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np

from PySDM import Builder
from PySDM.backends import CPU
from PySDM.environments import Box
from PySDM.physics import si
from PySDM.products.condensation import ActivableFraction


def test_critical_supersaturation():
    # arrange
    T = 300 * si.K
    n_sd = 100
    S_max = 0.01
    vdry = np.linspace(0.001, 1, n_sd) * si.um**3

    env = Box(dt=np.nan, dv=np.nan)
    builder = Builder(n_sd=n_sd, backend=CPU(), environment=env)
    env["T"] = T
    particulator = builder.build(
        attributes={
            "multiplicity": np.ones(n_sd),
            "volume": np.linspace(0.01, 10, n_sd) * si.um**3,
            "dry volume": vdry,
            "kappa times dry volume": 0.9 * vdry,
        },
        products=(ActivableFraction(),),
    )

    # act
    AF = particulator.products["activable fraction"].get(S_max=S_max)

    # assert
    assert 0 < AF < 1
