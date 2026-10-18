# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest

from PySDM import Particulator
from PySDM.backends import CPU
from PySDM.environments import Box
from PySDM.formulae import Formulae
from PySDM.physics.collision_kernel import golovin
from PySDM.backends.impl_common.indexed_storage import make_IndexedStorage
from PySDM.backends.impl_common.pair_indicator import make_PairIndicator
from PySDM.backends.impl_common.index import make_Index
from PySDM.backends.impl_common.pairwise_storage import make_PairwiseStorage

from PySDM.formulae import _choices
from PySDM.physics import collision_kernel
from PySDM.products import (
    LiquidWaterContent,
    ParticleConcentration,
)
from PySDM.physics import si, constants_defaults
from PySDM.physics.dimensional_analysis import DimensionalAnalysis


class TestKernels:
    @staticmethod
    @pytest.mark.parametrize(
        "x", [pytest.param(5e-10), pytest.param(np.full(10, 5e-10))]
    )
    def test_golovin_analytic_solution_underflow(x):
        # Arrange
        formulae = Formulae()
        b = 1.5e3
        x_0 = formulae.trivia.volume(radius=30.531e-6)
        N_0 = 2**23

        # Act
        value = golovin.analytic_solution(b=b, x=x, t=1200, x_0=x_0, N_0=N_0)

        # Assert
        assert np.all(np.isfinite(value))

    @staticmethod
    @pytest.mark.parametrize("c", (0.0, 1.0))
    def test_simple_geometric(c):
        # arrange
        volume = np.asarray([44.0, 666.0])

        particulator = Particulator(
            n_sd=volume.size,
            environment=Box(
                dv=None,
                dt=np.nan,
                backend=CPU(
                    Formulae(
                        collision_kernel="SimpleGeometric",
                        constants={"KERNEL_SIMPLE_GEOMETRIC_c": c},
                    )
                ),
            ),
            attributes={"volume": volume, "multiplicity": np.ones_like(volume)},
        )
        sut = particulator.collision_kernel

        _PairwiseStorage = particulator.PairwiseStorage
        _Indicator = particulator.PairIndicator
        output = _PairwiseStorage.from_ndarray(
            np.full(shape=volume.size // 2, fill_value=np.nan)
        )
        is_first_in_pair = _Indicator(length=volume.size)
        is_first_in_pair.indicator = particulator.Storage.from_ndarray(
            np.asarray([True, False])
        )
        # act
        sut(output=output, is_first_in_pair=is_first_in_pair)

        # assert
        if c > 0.0:
            np.testing.assert_array_less([0.0], output.to_ndarray())
        else:
            np.testing.assert_array_equal([0.0], output.to_ndarray())

    @staticmethod
    @pytest.mark.parametrize("volume", (np.array([1.0, 2.0]), np.array([1.0, 1.0])))
    def test_simple_geometric_same_size(volume):
        # arrange
        particulator = Particulator(
            n_sd=volume.size,
            environment=Box(
                dv=None,
                dt=np.nan,
                backend=CPU(
                    Formulae(
                        collision_kernel="SimpleGeometric",
                        constants={"KERNEL_SIMPLE_GEOMETRIC_c": 1.0},
                    )
                ),
            ),
            attributes={"volume": volume, "multiplicity": np.ones_like(volume)},
        )
        sut = particulator.collision_kernel

        _PairwiseStorage = particulator.PairwiseStorage
        _Indicator = particulator.PairIndicator
        output = _PairwiseStorage.from_ndarray(
            np.full(shape=volume.size // 2, fill_value=np.nan)
        )
        is_first_in_pair = _Indicator(length=volume.size)
        is_first_in_pair.indicator = particulator.Storage.from_ndarray(
            np.asarray([True, False])
        )

        # act
        sut(output=output, is_first_in_pair=is_first_in_pair)

        # assert
        if volume[0] == volume[1]:
            np.testing.assert_array_equal([0.0], output.to_ndarray())
        else:
            np.testing.assert_array_less([0.0], output.to_ndarray())

    @staticmethod
    @pytest.mark.parametrize(
        "radius",
        (
            (49e-6, 1e-6),
            (50e-6 + 1e-9, 1e-6),
        ),
    )
    def test_long1974_regimes(radius):
        # arrange
        formulae = Formulae(collision_kernel="Long1974")
        particulator = Particulator(
            attributes={
                "volume": formulae.trivia.volume(radius=np.asarray(radius)),
                "multiplicity": np.ones_like(radius),
            },
            n_sd=len(radius),
            environment=Box(
                dv=None,
                dt=np.nan,
                backend=CPU(formulae),
            ),
        )
        sut = particulator.collision_kernel

        _PairwiseStorage = particulator.PairwiseStorage
        _Indicator = particulator.PairIndicator
        output = _PairwiseStorage.from_ndarray(np.array([0.0]))
        is_first_in_pair = _Indicator(length=len(radius))
        is_first_in_pair.indicator = particulator.Storage.from_ndarray(
            np.asarray([True, False])
        )

        # act
        sut(output=output, is_first_in_pair=is_first_in_pair)

        radius_lg = np.max(radius)

        # assert
        if radius_lg < 50e-6:
            np.testing.assert_array_less(output.to_ndarray(), [3.0e-9])
        else:
            np.testing.assert_array_less([3.0e-9], output.to_ndarray())

    @staticmethod
    def test_multiphase():
        # arrange
        formulae = Formulae(
            particle_shape_and_density="MixedPhaseSpheres",
            collision_kernel="ConstantK+ConstantIce",
            constants={
                "KERNEL_CONSTANTK_a": 1,
                "KERNEL_CONSTANT_ICE_a": 2,
                "KERNEL_CONSTANT_ICE_b": 3,
            },
        )
        backend = CPU(formulae)

        signed_masses = (
            1,
            1,
            -1,
            1,
            -1,
            -1,
        )
        n_sd = len(signed_masses)
        n_pairs = n_sd // 2

        idx = make_Index(backend).identity_index(n_sd)
        dummy = make_IndexedStorage(backend).from_ndarray(
            idx, np.asarray([np.nan] * n_sd)
        )
        output = make_PairwiseStorage(backend).from_ndarray(
            np.asarray([np.nan] * n_pairs)
        )
        signed_water_mass = make_IndexedStorage(backend).from_ndarray(
            idx, np.asarray(signed_masses)
        )
        is_first_in_pair = make_PairIndicator(backend)(length=n_pairs)
        is_first_in_pair.indicator = backend.Storage.from_ndarray(
            np.asarray([True, False] * n_pairs)
        )

        # act
        backend.collision_kernel(
            attr0=dummy,
            attr1=dummy,
            attr2=dummy,
            signed_water_mass=signed_water_mass,
            output=output,
            is_first_in_pair=is_first_in_pair,
        )

        # assert
        assert (output.to_ndarray() == [1, 2, 3]).all()

    @staticmethod
    @pytest.mark.parametrize("variant", _choices(collision_kernel))
    def test_collision_kernel_call(backend_class, variant):
        if variant == "Null":
            pytest.skip()

        formulae = Formulae(
            collision_kernel=variant,
            seed=666,
            constants={
                "KERNEL_GOLOVIN_b": 0,
                "KERNEL_CONSTANTK_a": 1,
                "KERNEL_LINEAR_a": 1,
                "KERNEL_LINEAR_b": 1,
                "KERNEL_SIMPLE_GEOMETRIC_c": 1,
                "KERNEL_CONSTANT_ICE_a": 2,
                "KERNEL_CONSTANT_ICE_b": 3,
            },
        )
        env = Box(dt=1 * si.s, dv=1 * si.m**3, backend=backend_class(formulae=formulae))
        particulator = Particulator(
            n_sd=(n_sd := 2),
            environment=env,
            products=(LiquidWaterContent(name="qc"), ParticleConcentration(name="nc")),
            attributes={
                "multiplicity": np.ones(n_sd),
                "volume": np.ones(n_sd) * si.mm**3,
            },
        )

        # act
        nc_initial = particulator.products["nc"].get()
        particulator.advance(n_steps=1)

        # assert
        qc, nc = (particulator.products[k].get() for k in ("qc", "nc"))
        assert np.isfinite([qc, nc]).all() and qc > 0 and nc > 0
        assert nc <= nc_initial

    @staticmethod
    def test_golovin_units():
        with DimensionalAnalysis():
            si = constants_defaults.si  # pylint: disable=redefined-outer-name
            vol = 0 * si.m**3
            prob = golovin.Golovin.liquid_liquid(
                constants_defaults, vol, vol, np.nan, np.nan, np.nan, np.nan
            )
        assert prob.check("[volume] / [time]")
