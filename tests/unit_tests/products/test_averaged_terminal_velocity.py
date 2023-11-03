# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np

from PySDM import Builder
from PySDM.backends import CPU
from PySDM.environments import Box
from PySDM.physics import si
from PySDM.products.displacement import AveragedTerminalVelocity

T = 300 * si.K
dt = 1
r = (
    np.array([0.078, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6])
    * si.mm
    / 2
)
u = np.array([18, 27, 72, 117, 162, 206, 247, 287, 327, 367, 403, 464, 517, 565]) / 100


class TestAveragedTerminalVelocity:
    @staticmethod
    def _make_particulator(attributes: dict, weighting="volume"):
        env = Box(dt=dt, dv=np.nan)
        builder = Builder(
            n_sd=len(attributes["multiplicity"]), backend=CPU(), environment=env
        )
        env["T"] = T
        return builder.build(
            attributes=attributes,
            products=(AveragedTerminalVelocity(weighting=weighting),),
        )

    def test_mono_disperse(self):
        # arrange
        n_sd = 10
        vol = 4 / 3 * np.pi * 500**3 * si.um**3
        particulator = self._make_particulator(
            attributes={
                "multiplicity": np.ones(n_sd),
                "volume": np.full(n_sd, vol),
            }
        )

        # act
        vt = particulator.products["averaged terminal velocity"].get()[0]

        # assert
        assert (vt - 4.03) ** 2 < 1e-6

    def test_number_averaged_value(self):
        # arrange
        vol = 4 / 3 * np.pi * r**3
        particulator = self._make_particulator(
            attributes={
                "multiplicity": np.ones(len(r)),
                "volume": vol,
            },
            weighting="number",
        )

        # act
        vt = particulator.products["averaged terminal velocity"].get()[0]

        # assert
        assert (vt - np.mean(u)) ** 2 < 1e-6

    def test_volume_averaged_value(self):
        # arrange
        vol = 4 / 3 * np.pi * r**3
        particulator = self._make_particulator(
            attributes={
                "multiplicity": np.ones(len(r)),
                "volume": vol,
            },
            weighting="volume",
        )

        # act
        vt = particulator.products["averaged terminal velocity"].get()[0]

        # assert
        assert (vt - np.sum(vol * u) / np.sum(vol)) ** 2 < 1e-6
