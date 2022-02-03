# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest
from PySDM.backends import CPU
from PySDM.dynamics import Breakup
from PySDM.dynamics.collisions.kernels import ConstantK
from PySDM.environments import Box
from PySDM.physics.breakup_fragmentations import AlwaysN # TODO #743 move to dynamics
from PySDM import Builder
from PySDM.physics import si

@pytest.mark.parametrize("params", [
    {
        "attributes": {
            "n": np.asarray([1, 1]),
            "volume": np.asarray([100*si.um**3, 100*si.um**3])
        },
        "breakup": Breakup(ConstantK(1 / si.s), AlwaysN(2)),
        "nsteps": 10
    },
    ])
def test_breakup(params, backend_class = CPU):
    # Arrange
    n_sd = len(params["attributes"]["n"])
    builder = Builder(n_sd, backend_class())
    builder.set_environment(Box(dv=1*si.cm**3, dt=1*si.s))
    builder.add_dynamic(params["breakup"])
    particulator = builder.build(attributes = params["attributes"], products = ())

    # Act
    particulator.run(params["nsteps"])

    # Assert
    assert (particulator.attributes['n'].to_ndarray() > 0).all()
    assert (np.sum(particulator.attributes['n'].to_ndarray()) >= np.sum(params['attributes']['n']))
