import numpy as np
import pytest

from PySDM.dynamics import DropLocalThermodynamics, VapourDepositionOnIce
from PySDM import Builder, Formulae

from tests.unit_tests.dynamics.test_vapour_deposition_on_ice import MoistBox

# TODO: hello-world scenario: constant timescale, no diffusional growth, pure relaxation, assert that local temp relaxes to ambient according to the tau value


@pytest.mark.parametrize("tau", ())
def test_drop_local_thermodynamics(tau, backend_class, n_sd=1):
    # arrange
    formulae = Formulae(
        turbulent_relaxation_timescale="Constant", constants={"TODO": tau}
    )
    env = MoistBox(dt=np.nan, dv=np.nan)
    builder = Builder(n_sd=n_sd, backend=backend_class(formulae), environment=env)
    builder.add_dynamic(DropLocalThermodynamics())
    # builder.add_dynamic(VapourDepositionOnIce)
    particulator = builder.build(products=(), attributes={})

    # act
    particulator.run(steps=1)

    # assert
    assert something == f(tau)
