# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
from PySDM.products.freezing import CoolingRate
from PySDM import Builder
from PySDM.backends import CPU
from PySDM.environments import Box
from PySDM.physics import si


def test_cooling_rate():
    # arrange
    T = 300 * si.K
    n_sd = 100
    dt = 44
    dT = -2

    builder = Builder(n_sd=n_sd, backend=CPU())
    env = Box(dt=dt, dv=np.nan)
    builder.set_environment(env)
    env['T'] = T
    particulator = builder.build(
        attributes={
            'n': np.ones(n_sd),
            'volume': np.linspace(.01, 10, n_sd) * si.um**3
        },
        products=(CoolingRate(),)
    )

    #act & assert
    cr = particulator.products['cooling rate'].get()
    assert np.isnan(cr).all()

    particulator.run(1)
    particulator.attributes.mark_updated('cell id')
    cr = particulator.products['cooling rate'].get()
    assert (cr == 0).all()

    env['T'] += dT
    particulator.run(1)
    particulator.attributes.mark_updated('cell id')
    cr = particulator.products['cooling rate'].get()
    np.testing.assert_allclose(cr, dT / dt)
