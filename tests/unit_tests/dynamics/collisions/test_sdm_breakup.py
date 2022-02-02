# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
from turtle import back
import numpy as np
import pytest
from PySDM.backends import CPU
from PySDM.dynamics import Breakup
from PySDM.dynamics.collisions.kernels import ConstantK
from PySDM.environments import Box
from PySDM.physics.breakup_fragmentations import AlwaysN # TODO #743 move breakup fragmentations from physics to dynamics
from PySDM import Builder
from PySDM.physics import si

@pytest.mark.parametrize("input", [
    {"attributes": {"n": np.asarray([1, 1]), "volume": np.asarray([100*si.um**3, 100*si.um**3])}, 
     "breakup": Breakup(ConstantK(1 / si.s), AlwaysN(2)), "nsteps": 10},
    # {"multiplicities": [5, 2], "breakup": Breakup(ConstantK(), AlwaysN())}
    ])
def test_breakup(input, backend_class = CPU):
    # Arrange
    n_sd = len(input["attributes"]["n"])
    builder = Builder(n_sd, backend_class())
    builder.set_environment(Box(dv=1*si.cm**3, dt=1*si.s))
    builder.add_dynamic(input["breakup"])
    particulator = builder.build(attributes = input["attributes"], products = ())

    # Act
    particulator.run(input["nsteps"])

    # Assert
    assert (particulator.attributes['n'].to_ndarray() > 0).all()
    assert (np.sum(particulator.attributes['n'].to_ndarray()) >= np.sum(input['attributes']['n']))