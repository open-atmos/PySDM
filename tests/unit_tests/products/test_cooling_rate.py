# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np

from PySDM import Builder
from PySDM.backends import CPU
from PySDM.environments import Box
from PySDM.physics import si
from PySDM.products.freezing import CoolingRate

T = 300 * si.K
n_sd = 100
dt = 44
dT = -2


class TestCoolingRate:
    @staticmethod
    def _make_particulator():
        env = Box(dt=dt, dv=np.nan)
        builder = Builder(n_sd=n_sd, backend=CPU(), environment=env)
        env["T"] = T
        return builder.build(
            attributes={
                "multiplicity": np.ones(n_sd),
                "volume": np.linspace(0.01, 10, n_sd) * si.um**3,
            },
            products=(CoolingRate(),),
        )

    def test_nan_at_t_zero(self):
        # arrange
        particulator = self._make_particulator()

        # act
        cr = particulator.products["cooling rate"].get()

        # assert
        assert np.isnan(cr).all()

    def test_zero_with_no_env_change(self):
        # arrange
        particulator = self._make_particulator()

        # act
        particulator.run(1)
        particulator.attributes.mark_updated("cell id")
        cr = particulator.products["cooling rate"].get()

        # assert
        assert (cr == 0).all()

    def test_with_env_change(self):
        # arrange
        particulator = self._make_particulator()

        # act
        particulator.run(1)
        particulator.environment["T"] += dT
        particulator.attributes.mark_updated("cell id")
        cr = particulator.products["cooling rate"].get()

        # assert
        np.testing.assert_allclose(actual=cr, desired=-dT / dt)
