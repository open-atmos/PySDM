# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import math
from collections import namedtuple

import numpy as np

import pytest

from PySDM import Builder
from PySDM.backends import CPU
from PySDM.environments import Box, Kinematic1D
from PySDM.physics import si
from PySDM.products.freezing import CoolingRate
from PySDM.impl.mesh import Mesh
from PySDM.dynamics import Displacement, AmbientThermodynamics

T = 300 * si.K
n_sd = 100
dt = 44
dT = -2


class TestCoolingRate:
    @staticmethod
    def _make_particulator():
        env = Box(dt=dt, dv=np.nan)
        builder = Builder(n_sd=n_sd, backend=CPU(), environment=env)
        particulator = builder.build(
            attributes={
                "multiplicity": np.ones(n_sd),
                "volume": np.linspace(0.01, 10, n_sd) * si.um**3,
            },
            products=(CoolingRate(),),
        )
        particulator.environment["T"] = T
        return particulator

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

    @staticmethod
    @pytest.mark.parametrize("courant_number", (0.2, 0.8))
    @pytest.mark.parametrize(
        "timestep",
        (
            0.5 * si.s,
            5 * si.s,
        ),
    )
    def test_single_column_constant_updraft(  # pylint: disable=too-many-locals
        *,
        courant_number,
        timestep,
        mean_n_sd_per_gridbox=1000,
        nz=10,
        z_max=1 * si.km,
        signed_thd_lapse_rate=-5 * si.K / si.km,
        constant_rhod=1 * si.kg / si.m**3,
        n_steps=3,
    ):
        # arrange
        builder = Builder(
            environment=Kinematic1D(
                dt=timestep,
                mesh=Mesh(grid=(nz,), size=(z_max,)),
                thd_of_z=lambda z: signed_thd_lapse_rate * z + 300 * si.K,
                rhod_of_z=lambda z: 0 * z + constant_rhod,
            ),
            n_sd=mean_n_sd_per_gridbox * nz,
            backend=CPU(),
        )
        builder.add_dynamic(AmbientThermodynamics())

        class EulerianAdvection:
            solvers = namedtuple(typename="_", field_names=("advectee",))(
                advectee=namedtuple(typename="__", field_names=("ravel", "shape"))(
                    ravel=lambda: None, shape=(nz,)
                )
            )

            def instantiate(self, *, builder):
                assert builder
                return self

            def __call__(self):
                pass

        builder.add_dynamic(EulerianAdvection())
        builder.add_dynamic(Displacement())

        cellular_attributes = {}
        (
            cellular_attributes["cell id"],
            cellular_attributes["cell origin"],
            cellular_attributes["position in cell"],
        ) = builder.particulator.environment.mesh.cellular_attributes(
            positions=np.random.random_sample(size=builder.particulator.n_sd).reshape(
                (1, -1)
            )
            * nz
        )
        particulator = builder.build(
            attributes={
                "multiplicity": np.ones(builder.particulator.n_sd),
                "water mass": np.ones(builder.particulator.n_sd) * 1 * si.ug,
                **cellular_attributes,
            },
            products=(CoolingRate(),),
        )
        particulator.dynamics["Displacement"].upload_courant_field(
            courant_field=(np.full(nz + 1, fill_value=courant_number),)
        )

        # act
        particulator.run(steps=n_steps)

        cooling_rates = particulator.products["cooling rate"].get()

        # assert
        assert (
            0.5 * particulator.n_sd
            < len(particulator.attributes["cell id"])
            < particulator.n_sd
        )
        assert len(cooling_rates) == nz

        valid_cells = slice(math.ceil(courant_number * n_steps), None)

        dz = z_max / nz
        dz_dt = courant_number * dz / timestep
        mean_temperature_lapse_rate = (
            np.mean(np.diff(particulator.environment["T"].to_ndarray())) / dz
        )

        np.testing.assert_allclose(
            actual=cooling_rates[valid_cells],
            desired=-1 * mean_temperature_lapse_rate * dz_dt,
            rtol=0.2,
        )
