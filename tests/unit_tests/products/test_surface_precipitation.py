"""sanity checks for the surface precipitation product"""

import pytest
import numpy as np

from PySDM import Builder
from PySDM.backends import GPU
from PySDM.physics import si
from PySDM.impl.mesh import Mesh
from PySDM.environments import Box, Kinematic1D
from PySDM.products import SurfacePrecipitation
from PySDM.dynamics import Displacement


class TestSurfacePrecipitation:
    @staticmethod
    def test_fails_for_0d_env(backend_instance):
        """checks if builder fails with a relevant error message on an attempt to use surface
        precipitation product with a zero-dimensional environment"""
        # arrange
        builder = Builder(
            n_sd=1, backend=backend_instance, environment=Box(dt=np.nan, dv=np.nan)
        )
        builder.add_dynamic(Displacement())

        # act
        with pytest.raises(AssertionError) as ex:
            _ = builder.build(attributes={}, products=(SurfacePrecipitation(),))

        # assert
        assert "n_dims > 0" in str(ex.traceback[-1].statement)

    @staticmethod
    @pytest.mark.parametrize(
        "env_class, env_ctor_args",
        (
            pytest.param(
                Kinematic1D,
                {
                    "mesh": Mesh(grid=(1,), size=(44 * si.m,)),
                    "thd_of_z": lambda z: z * np.nan,
                    "rhod_of_z": lambda z: z * np.nan,
                },
                id=Kinematic1D.__name__,
            ),
        ),
    )
    @pytest.mark.parametrize("dt", (0.1 * si.s, 10 * si.s))
    @pytest.mark.parametrize("drop_mass", (1.2345 * si.ug,))
    @pytest.mark.parametrize("multiplicity", (1, 2, 3))
    @pytest.mark.parametrize("n_sd", (1, 44, 666))
    # TODO #1418 add tests for counting_level
    def test_surface_precipitation(
        *, env_class, env_ctor_args, backend_instance, dt, drop_mass, multiplicity, n_sd
    ):
        """uses a monodisperse super-droplet setup to check if reported precip
        matches drop and flow params, the droplet is initialised at position z=0,
        so any downward movement triggers counting as precip
        """

        if isinstance(backend_instance, GPU):
            pytest.skip("TODO #1418")

        # arrange
        n_cell = 1
        builder = Builder(
            n_sd=n_sd,
            backend=backend_instance,
            environment=env_class(**env_ctor_args, dt=dt),
        )
        builder.add_dynamic(Displacement(enable_sedimentation=True))
        particulator = builder.build(
            attributes={
                "multiplicity": np.asarray([multiplicity] * n_sd),
                "water mass": np.asarray([drop_mass] * n_sd),
                "cell id": np.zeros(n_sd, dtype=int),
                "cell origin": np.zeros(n_sd, dtype=int),
                "position in cell": np.zeros(n_sd),
            },
            products=(SurfacePrecipitation(),),
        )
        particulator.dynamics[Displacement.__name__].upload_courant_field(
            courant_field=(np.zeros(n_cell + 1),)
        )

        # act
        particulator.run(steps=1)

        # assert
        water_volume = (
            n_sd * multiplicity * drop_mass / particulator.formulae.constants.rho_w
        )
        np.testing.assert_approx_equal(
            actual=particulator.products["surface precipitation"].get(),
            desired=water_volume
            / particulator.environment.mesh.domain_bottom_surface_area
            / dt,
        )

        # act again
        particulator.run(steps=1)

        # assert again
        assert particulator.products["surface precipitation"].get() == 0
