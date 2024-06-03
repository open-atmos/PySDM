# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest

import PySDM.physics.constants as const
from PySDM import Builder, Formulae
from PySDM.backends import CPU
from PySDM.backends.impl_common.pair_indicator import make_PairIndicator
from PySDM.dynamics import Breakup
from PySDM.dynamics.collisions import breakup_fragmentations
from PySDM.dynamics.collisions.breakup_efficiencies import ConstEb
from PySDM.dynamics.collisions.breakup_fragmentations import AlwaysN
from PySDM.dynamics.collisions.coalescence_efficiencies import ConstEc
from PySDM.dynamics.collisions.collision import DEFAULTS, Collision
from PySDM.dynamics.collisions.collision_kernels import ConstantK, Geometric
from PySDM.environments import Box
from PySDM.initialisation import spectra
from PySDM.initialisation.sampling.spectral_sampling import ConstantMultiplicity
from PySDM.physics import si
from PySDM.physics.trivia import Trivia
from PySDM.products.size_spectral import ParticleVolumeVersusRadiusLogarithmSpectrum


def volume_to_mass(particulator, volume):
    if isinstance(volume, (list, tuple)):
        return list(
            map(particulator.formulae.particle_shape_and_density.volume_to_mass, volume)
        )

    raise NotImplementedError


class TestSDMBreakup:
    @staticmethod
    @pytest.mark.parametrize(
        "dt",
        [
            1 * si.s,
            10 * si.s,
        ],
    )
    def test_nonadaptive_same_results_regardless_of_dt(dt, backend_class):
        # Arrange
        attributes = {
            "multiplicity": np.asarray([1, 1]),
            "volume": np.asarray([100 * si.um**3, 100 * si.um**3]),
        }
        breakup = Breakup(
            collision_kernel=ConstantK(1 * si.cm**3 / si.s),
            fragmentation_function=AlwaysN(4),
            adaptive=False,
            warn_overflows=False,
        )
        nsteps = 10

        n_sd = len(attributes["multiplicity"])
        env = Box(dv=1 * si.cm**3, dt=dt)
        builder = Builder(
            n_sd,
            backend_class(Formulae(fragmentation_function="AlwaysN")),
            environment=env,
        )
        builder.add_dynamic(breakup)
        particulator = builder.build(attributes=attributes, products=())

        # Act
        particulator.run(nsteps)

        # Assert
        assert (particulator.attributes["multiplicity"].to_ndarray() > 0).all()
        assert (
            particulator.attributes["multiplicity"].to_ndarray()
            != attributes["multiplicity"]
        ).any()
        assert np.sum(particulator.attributes["multiplicity"].to_ndarray()) >= np.sum(
            attributes["multiplicity"]
        )
        assert (
            particulator.attributes["multiplicity"].to_ndarray()
            == np.array([1024, 1024])
        ).all()

    @staticmethod
    @pytest.mark.parametrize(
        "params",
        [
            {
                "gamma": 1.0,
                "rand": 1.0,
            },
            {"gamma": 1.0, "rand": 0.1},
            pytest.param(
                {"gamma": 1.0, "rand": 0.0}, marks=pytest.mark.xfail(strict=True)
            ),
        ],
    )
    def test_single_collision_bounce(params, backend_instance):
        # Arrange
        backend = backend_instance
        n_sd = 2
        env = Box(dv=np.NaN, dt=np.NaN)
        builder = Builder(n_sd, backend, environment=env)
        n_init = [6, 6]
        particulator = builder.build(
            attributes={
                "multiplicity": np.asarray(n_init),
                "volume": np.asarray([100 * si.um**3, 100 * si.um**3]),
            },
            products=(),
        )

        pairwise_zeros = particulator.PairwiseStorage.from_ndarray(np.array([0.0]))
        general_zeros = particulator.Storage.from_ndarray(np.array([0]))

        gamma = particulator.PairwiseStorage.from_ndarray(np.asarray([params["gamma"]]))
        rand = particulator.PairwiseStorage.from_ndarray(np.asarray([params["rand"]]))
        fragment_mass = particulator.PairwiseStorage.from_ndarray(
            np.array([50 * si.um**3], dtype=float)
        )
        is_first_in_pair = make_PairIndicator(backend)(n_sd)

        # Act
        particulator.collision_coalescence_breakup(
            enable_breakup=True,
            gamma=gamma,
            rand=rand,
            Ec=pairwise_zeros,
            Eb=pairwise_zeros,
            fragment_mass=fragment_mass,
            coalescence_rate=general_zeros,
            breakup_rate=general_zeros,
            breakup_rate_deficit=general_zeros,
            is_first_in_pair=is_first_in_pair,
            warn_overflows=False,
            max_multiplicity=DEFAULTS.max_multiplicity,
        )

        # Assert
        assert (particulator.attributes["multiplicity"].to_ndarray() == n_init).all()

    @staticmethod
    @pytest.mark.parametrize(
        "params",
        [
            {
                "gamma": 1.0,
                "rand": 1.0,
                "Eb": 1.0,
                "n_init": [1, 1],
                "is_first_in_pair": [True, False],
            },
            {
                "gamma": 1.0,
                "rand": 1.0,
                "Eb": 1.0,
                "n_init": [2, 1],
                "is_first_in_pair": [True, False],
            },
            {
                "gamma": 1.0,
                "rand": 1.0,
                "Eb": 1.0,
                "n_init": [2, 1, 2],
                "is_first_in_pair": [True, False, False],
            },
            {
                "gamma": 1.0,
                "rand": 1.0,
                "Eb": 1.0,
                "n_init": [2, 1, 2, 1],
                "is_first_in_pair": [True, False, True, False],
            },
        ],
    )
    def test_breakup_counters(
        params, backend_instance
    ):  # pylint: disable=too-many-locals
        # Arrange
        n_init = params["n_init"]
        n_sd = len(n_init)
        env = Box(dv=np.NaN, dt=np.NaN)
        builder = Builder(n_sd, backend_instance, environment=env)
        particulator = builder.build(
            attributes={
                "multiplicity": np.asarray(n_init),
                "volume": np.asarray([100 * si.um**3] * n_sd),
            },
            products=(),
        )

        n_pairs = n_sd // 2
        pairwise_zeros = particulator.PairwiseStorage.from_ndarray(
            np.array([0.0] * n_pairs)
        )
        general_zeros = particulator.Storage.from_ndarray(np.array([0] * n_sd))

        gamma = particulator.PairwiseStorage.from_ndarray(
            np.array([params["gamma"]] * n_pairs)
        )
        rand = particulator.PairwiseStorage.from_ndarray(
            np.array([params["rand"]] * n_pairs)
        )
        Eb = particulator.PairwiseStorage.from_ndarray(
            np.array([params["Eb"]] * n_pairs)
        )
        breakup_rate = particulator.Storage.from_ndarray(np.array([0]))
        frag_mass = particulator.PairwiseStorage.from_ndarray(
            np.array(volume_to_mass(particulator, [2 * si.m**3]) * n_pairs)
        )
        is_first_in_pair = particulator.PairIndicator(n_sd)
        is_first_in_pair.indicator[:] = particulator.Storage.from_ndarray(
            np.asarray(params["is_first_in_pair"])
        )

        # Act
        particulator.collision_coalescence_breakup(
            enable_breakup=True,
            gamma=gamma,
            rand=rand,
            Ec=pairwise_zeros,
            Eb=Eb,
            fragment_mass=frag_mass,
            coalescence_rate=general_zeros,
            breakup_rate=breakup_rate,
            breakup_rate_deficit=general_zeros,
            is_first_in_pair=is_first_in_pair,
            warn_overflows=False,
            max_multiplicity=DEFAULTS.max_multiplicity,
        )

        # Assert
        cell_id = 0
        assert breakup_rate.to_ndarray()[cell_id] == np.sum(
            params["gamma"] * get_smaller_of_pairs(is_first_in_pair, n_init)
        )

    @staticmethod
    @pytest.mark.parametrize(
        "params",
        [
            {
                "gamma": [1.0],
                "n_init": [1, 1],
                "v_init": [1, 1],
                "n_expected": [2, 2],
                "v_expected": [0.5, 0.5],
                "expected_deficit": [0.0],
                "is_first_in_pair": [True, False],
                "frag_volume": [0.5],
            },
            {
                "gamma": [2.0],
                "n_init": [20, 4],
                "v_init": [1, 2],
                "n_expected": [4, 24],
                "v_expected": [1, 1],
                "expected_deficit": [0.0],
                "is_first_in_pair": [True, False],
                "frag_volume": [1.0],
            },
            {
                "gamma": [2.0],
                "n_init": [1, 1],
                "v_init": [1, 1],
                "n_expected": [2, 2],
                "v_expected": [0.5, 0.5],
                "expected_deficit": [1.0],
                "is_first_in_pair": [True, False],
                "frag_volume": [0.5],
            },
            {
                "gamma": [2.0],
                "n_init": [3, 1],
                "v_init": [1, 1],
                "n_expected": [2, 4],
                "v_expected": [1.0, 0.5],
                "expected_deficit": [1.0],
                "is_first_in_pair": [True, False],
                "frag_volume": [0.5],
            },
            {
                "gamma": [2.0],
                "n_init": [9, 2],
                "v_init": [1, 2],
                "n_expected": [1, 12],
                "v_expected": [1, 1],
                "expected_deficit": [0.0],
                "is_first_in_pair": [True, False],
                "frag_volume": [1.0],
            },
            {
                "gamma": [1.0],
                "n_init": [12, 1],
                "v_init": [1, 1],
                "n_expected": [11, 2],
                "v_expected": [1, 1],
                "expected_deficit": [0.0],
                "is_first_in_pair": [True, False],
                "frag_volume": [1.0],
            },
            {
                "gamma": [1.0],
                "n_init": [15, 2],
                "v_init": [2, 6],
                "n_expected": [13, 4],
                "v_expected": [2, 4],
                "expected_deficit": [0.0],
                "is_first_in_pair": [True, False],
                "frag_volume": [4],
            },
            {
                "gamma": [1.0],
                "n_init": [13, 4],
                "v_init": [2, 4],
                "n_expected": [9, 6],
                "v_expected": [2, 4],
                "expected_deficit": [0.0],
                "is_first_in_pair": [True, False],
                "frag_volume": [4],
            },
            {
                "gamma": [3.0],
                "n_init": [15, 2],
                "v_init": [2, 6],
                "n_expected": [3, 9],
                "v_expected": [2, 4],
                "expected_deficit": [0.0],
                "is_first_in_pair": [True, False],
                "frag_volume": [4],
            },
            {
                "gamma": [0.0],
                "n_init": [15, 2],
                "v_init": [2, 6],
                "n_expected": [15, 2],
                "v_expected": [2, 6],
                "expected_deficit": [0.0],
                "is_first_in_pair": [True, False],
                "frag_volume": [4],
            },
        ],
    )
    @pytest.mark.parametrize("flag", ("multiplicity", "v", "conserve", "deficit"))
    def test_attribute_update_single_breakup(
        params, flag, backend_class
    ):  # pylint: disable=too-many-locals
        # Arrange
        n_init = params["n_init"]
        n_sd = len(n_init)
        env = Box(dv=np.NaN, dt=np.NaN)
        builder = Builder(n_sd, backend_class(double_precision=True), environment=env)
        particulator = builder.build(
            attributes={
                "multiplicity": np.asarray(n_init),
                "volume": np.asarray(params["v_init"]),
            },
            products=(),
        )

        n_pairs = n_sd // 2
        rand = [1.0] * n_pairs
        Eb = [1.0] * n_pairs
        pairwise_zeros = particulator.PairwiseStorage.from_ndarray(
            np.array([0.0] * n_pairs)
        )
        general_zeros = particulator.Storage.from_ndarray(np.array([0] * n_sd))

        gamma = particulator.PairwiseStorage.from_ndarray(np.array(params["gamma"]))
        rand = particulator.PairwiseStorage.from_ndarray(np.array(rand))
        Eb = particulator.PairwiseStorage.from_ndarray(np.array(Eb))
        breakup_rate = particulator.Storage.from_ndarray(np.array([0]))
        breakup_rate_deficit = particulator.Storage.from_ndarray(np.array([0]))

        frag_mass = volume_to_mass(particulator, params["frag_volume"])
        frag_mass = particulator.PairwiseStorage.from_ndarray(
            np.asarray(frag_mass, dtype=float)
        )
        is_first_in_pair = particulator.PairIndicator(n_sd)
        is_first_in_pair.indicator[:] = particulator.Storage.from_ndarray(
            np.asarray(params["is_first_in_pair"], dtype=bool)
        )

        # Act
        particulator.collision_coalescence_breakup(
            enable_breakup=True,
            gamma=gamma,
            rand=rand,
            Ec=pairwise_zeros,
            Eb=Eb,
            fragment_mass=frag_mass,
            coalescence_rate=general_zeros,
            breakup_rate=breakup_rate,
            breakup_rate_deficit=breakup_rate_deficit,
            is_first_in_pair=is_first_in_pair,
            warn_overflows=False,
            max_multiplicity=DEFAULTS.max_multiplicity,
        )

        # Assert
        {
            "multiplicity": lambda: np.testing.assert_array_equal(
                particulator.attributes["multiplicity"].to_ndarray(),
                np.array(params["n_expected"]),
            ),
            "v": lambda: np.testing.assert_array_almost_equal(
                particulator.attributes["volume"].to_ndarray(),
                np.array(params["v_expected"]),
            ),
            "conserve": lambda: np.testing.assert_almost_equal(
                np.sum(
                    particulator.attributes["multiplicity"].to_ndarray()
                    * particulator.attributes["volume"].to_ndarray()
                ),
                np.sum(np.array(params["n_init"]) * np.array(params["v_init"])),
            ),
            "deficit": lambda: np.testing.assert_almost_equal(
                breakup_rate_deficit.to_ndarray(), np.array(params["expected_deficit"])
            ),
        }[flag]()

    @staticmethod
    @pytest.mark.parametrize("_n", [1, 2, 3, 4, 5])
    @pytest.mark.parametrize(
        "params",
        [
            {
                "n_init": [64, 2],
                "v_init": [128, 128],
                "is_first_in_pair": [True, False],
                "frag_volume": [128],
            },
            {
                "n_init": [20, 4],
                "v_init": [1, 2],
                "is_first_in_pair": [True, False],
                "frag_volume": [1.0],
            },
            {
                "n_init": [3, 1],
                "v_init": [1, 1],
                "is_first_in_pair": [True, False],
                "frag_volume": [0.5],
            },
            {
                "n_init": [64, 2],
                "v_init": [8, 16],
                "is_first_in_pair": [True, False],
                "n_fragment": [6],
                "frag_volume": [4.0],
            },
            {
                "n_init": [64, 2],
                "v_init": [6, 2],
                "is_first_in_pair": [True, False],
                "n_fragment": [2],
                "frag_volume": [4.0],
            },
        ],
    )
    def test_attribute_update_n_breakups(
        _n, params, backend_class=CPU
    ):  # pylint: disable=too-many-locals
        # Arrange

        assert len(params["frag_volume"]) == 1

        def run_simulation(_n_times, _gamma):
            n_init = params["n_init"]
            n_sd = len(n_init)
            env = Box(dv=np.NaN, dt=np.NaN)
            builder = Builder(n_sd, backend_class(), environment=env)
            particulator = builder.build(
                attributes={
                    "multiplicity": np.asarray(n_init),
                    "volume": np.asarray(params["v_init"]),
                },
                products=(),
            )

            n_pairs = n_sd // 2
            rand = [1.0] * n_pairs
            Eb = [1.0] * n_pairs
            pairwise_zeros = particulator.PairwiseStorage.from_ndarray(
                np.array([0.0] * n_pairs)
            )
            general_zeros = particulator.Storage.from_ndarray(np.array([0] * n_sd))

            gamma = particulator.PairwiseStorage.from_ndarray(np.array(_gamma))
            rand = particulator.PairwiseStorage.from_ndarray(np.array(rand))
            Eb = particulator.PairwiseStorage.from_ndarray(np.array(Eb))
            breakup_rate = particulator.Storage.from_ndarray(np.array([0]))
            breakup_rate_deficit = particulator.Storage.from_ndarray(np.array([0]))
            frag_mass = particulator.PairwiseStorage.from_ndarray(
                np.asarray(
                    volume_to_mass(particulator, params["frag_volume"]), dtype=float
                )
            )
            is_first_in_pair = particulator.PairIndicator(n_sd)
            is_first_in_pair.indicator[:] = particulator.Storage.from_ndarray(
                np.asarray(params["is_first_in_pair"], dtype=bool)
            )

            # Act
            for _ in range(_n_times):
                particulator.collision_coalescence_breakup(
                    enable_breakup=True,
                    gamma=gamma,
                    rand=rand,
                    Ec=pairwise_zeros,
                    Eb=Eb,
                    fragment_mass=frag_mass,
                    coalescence_rate=general_zeros,
                    breakup_rate=breakup_rate,
                    breakup_rate_deficit=breakup_rate_deficit,
                    is_first_in_pair=is_first_in_pair,
                    warn_overflows=False,
                    max_multiplicity=DEFAULTS.max_multiplicity,
                )

            res_mult = particulator.attributes["multiplicity"].to_ndarray()
            res_volume = particulator.attributes["volume"].to_ndarray()
            return res_mult, res_volume

        run1 = run_simulation(_n_times=1, _gamma=[_n])
        run2 = run_simulation(_n_times=_n, _gamma=[1])

        # Assert
        np.testing.assert_array_almost_equal(
            run1[0],
            run2[0],
        )
        np.testing.assert_array_almost_equal(
            run1[1],
            run2[1],
        )

    @staticmethod
    def test_multiplicity_overflow(backend=CPU()):  # pylint: disable=too-many-locals
        # Arrange
        params = {
            "gamma": [1.0],
            "n_init": [1, 3],
            "v_init": [1, 1],
            "is_first_in_pair": [True, False],
            "frag_volume": [2e-10],
        }
        n_init = params["n_init"]
        n_sd = len(n_init)
        env = Box(dv=np.NaN, dt=np.NaN)
        builder = Builder(n_sd, backend, environment=env)
        particulator = builder.build(
            attributes={
                "multiplicity": np.asarray(n_init),
                "volume": np.asarray(params["v_init"]),
            },
            products=(),
        )

        n_pairs = n_sd // 2
        rand = [1.0] * n_pairs
        Eb = [1.0] * n_pairs
        pairwise_zeros = particulator.PairwiseStorage.from_ndarray(
            np.array([0.0] * n_pairs)
        )
        general_zeros = particulator.Storage.from_ndarray(np.array([0] * n_sd))

        gamma = particulator.PairwiseStorage.from_ndarray(np.array(params["gamma"]))
        rand = particulator.PairwiseStorage.from_ndarray(np.array(rand))
        Eb = particulator.PairwiseStorage.from_ndarray(np.array(Eb))
        breakup_rate = particulator.Storage.from_ndarray(np.array([0]))
        breakup_rate_deficit = particulator.Storage.from_ndarray(np.array([0]))
        frag_mass = particulator.PairwiseStorage.from_ndarray(
            np.array(volume_to_mass(particulator, params["frag_volume"]), dtype=float)
        )
        is_first_in_pair = particulator.PairIndicator(n_sd)
        is_first_in_pair.indicator[:] = particulator.Storage.from_ndarray(
            np.asarray(params["is_first_in_pair"])
        )

        # Act
        particulator.collision_coalescence_breakup(
            enable_breakup=True,
            gamma=gamma,
            rand=rand,
            Ec=pairwise_zeros,
            Eb=Eb,
            fragment_mass=frag_mass,
            coalescence_rate=general_zeros,
            breakup_rate=breakup_rate,
            breakup_rate_deficit=breakup_rate_deficit,
            is_first_in_pair=is_first_in_pair,
            warn_overflows=True,
            max_multiplicity=DEFAULTS.max_multiplicity,
        )
        assert breakup_rate_deficit[0] > 0
        np.testing.assert_almost_equal(
            np.sum(
                particulator.attributes["multiplicity"].to_ndarray()
                * particulator.attributes["volume"].to_ndarray()
            ),
            np.sum(np.array(params["n_init"]) * np.array(params["v_init"])),
        )

    @staticmethod
    def test_same_multiplicity_overflow_no_substeps(
        backend=CPU(),
    ):  # pylint: disable=too-many-locals
        # Arrange
        params = {
            "gamma": [46.0],
            "n_init": [1, 1],
            "v_init": [1, 1],
            "is_first_in_pair": [True, False],
            "frag_volume": [0.5],
        }
        n_init = params["n_init"]
        n_sd = len(n_init)
        env = Box(dv=np.NaN, dt=np.NaN)
        builder = Builder(n_sd, backend, environment=env)
        particulator = builder.build(
            attributes={
                "multiplicity": np.asarray(n_init),
                "volume": np.asarray(params["v_init"]),
            },
            products=(),
        )

        n_pairs = n_sd // 2
        rand = [1.0] * n_pairs
        Eb = [1.0] * n_pairs
        pairwise_zeros = particulator.PairwiseStorage.from_ndarray(
            np.array([0.0] * n_pairs)
        )
        general_zeros = particulator.Storage.from_ndarray(np.array([0] * n_sd))

        gamma = particulator.PairwiseStorage.from_ndarray(np.array(params["gamma"]))
        rand = particulator.PairwiseStorage.from_ndarray(np.array(rand))
        Eb = particulator.PairwiseStorage.from_ndarray(np.array(Eb))
        breakup_rate = particulator.Storage.from_ndarray(np.array([0]))
        breakup_rate_deficit = particulator.Storage.from_ndarray(np.array([0]))
        frag_mass = particulator.PairwiseStorage.from_ndarray(
            np.array(volume_to_mass(particulator, params["frag_volume"]), dtype=float)
        )
        is_first_in_pair = particulator.PairIndicator(n_sd)
        is_first_in_pair.indicator[:] = particulator.Storage.from_ndarray(
            np.asarray(params["is_first_in_pair"])
        )

        # Act
        particulator.collision_coalescence_breakup(
            enable_breakup=True,
            gamma=gamma,
            rand=rand,
            Ec=pairwise_zeros,
            Eb=Eb,
            fragment_mass=frag_mass,
            coalescence_rate=general_zeros,
            breakup_rate=breakup_rate,
            breakup_rate_deficit=breakup_rate_deficit,
            is_first_in_pair=is_first_in_pair,
            warn_overflows=True,
            max_multiplicity=DEFAULTS.max_multiplicity,
        )
        assert breakup_rate_deficit[0] > 0
        np.testing.assert_equal(
            np.sum(
                particulator.attributes["multiplicity"].to_ndarray()
                * particulator.attributes["volume"].to_ndarray()
            ),
            np.sum(np.array(params["n_init"]) * np.array(params["v_init"])),
        )

    @staticmethod
    @pytest.mark.parametrize(
        "params",
        [
            {
                "gamma": [1.0],
                "n_init": [1, 1],
                "v_init": [1, 1],
                "n_expected": [1, 1],
                "v_expected": [1, 1],
                "expected_deficit": [0.0],
                "is_first_in_pair": [True, False],
                "frag_volume": [1.25],
            },
            {
                "gamma": [1.0],
                "n_init": [1, 1],
                "v_init": [1, 1],
                "n_expected": [1, 1],
                "v_expected": [1, 1],
                "expected_deficit": [0.0],
                "is_first_in_pair": [True, False],
                "frag_volume": [1 / 1.3],
            },
            {
                "gamma": [2.0],
                "n_init": [2, 1],
                "v_init": [1, 1],
                "n_expected": [1, 3],
                "v_expected": [1, 2 / 3],
                "expected_deficit": [1.0],
                "is_first_in_pair": [True, False],
                "frag_volume": [1 / 1.4],
            },
        ],
    )
    @pytest.mark.parametrize("flag", ("multiplicity", "v", "conserve", "deficit"))
    def test_noninteger_fragments(
        params, flag, backend_instance
    ):  # pylint: disable=too-many-locals
        # Arrange
        n_init = params["n_init"]
        n_sd = len(n_init)
        env = Box(dv=np.NaN, dt=np.NaN)
        builder = Builder(n_sd, backend_instance, environment=env)
        particulator = builder.build(
            attributes={
                "multiplicity": np.asarray(n_init),
                "volume": np.asarray(params["v_init"]),
            },
            products=(),
        )

        n_pairs = n_sd // 2
        rand = [1.0] * n_pairs
        Eb = [1.0] * n_pairs
        pairwise_zeros = particulator.PairwiseStorage.from_ndarray(
            np.array([0.0] * n_pairs)
        )
        general_zeros = particulator.Storage.from_ndarray(np.array([0] * n_sd))

        gamma = particulator.PairwiseStorage.from_ndarray(np.array(params["gamma"]))
        rand = particulator.PairwiseStorage.from_ndarray(np.array(rand))
        Eb = particulator.PairwiseStorage.from_ndarray(np.array(Eb))
        breakup_rate = particulator.Storage.from_ndarray(np.array([0]))
        breakup_rate_deficit = particulator.Storage.from_ndarray(np.array([0]))
        frag_mass = particulator.PairwiseStorage.from_ndarray(
            np.asarray(volume_to_mass(particulator, params["frag_volume"]), dtype=float)
        )
        is_first_in_pair = particulator.PairIndicator(n_sd)
        is_first_in_pair.indicator[:] = particulator.Storage.from_ndarray(
            np.asarray(params["is_first_in_pair"], dtype=bool)
        )

        # Act
        particulator.collision_coalescence_breakup(
            enable_breakup=True,
            gamma=gamma,
            rand=rand,
            Ec=pairwise_zeros,
            Eb=Eb,
            fragment_mass=frag_mass,
            coalescence_rate=general_zeros,
            breakup_rate=breakup_rate,
            breakup_rate_deficit=breakup_rate_deficit,
            is_first_in_pair=is_first_in_pair,
            warn_overflows=False,
            max_multiplicity=DEFAULTS.max_multiplicity,
        )

        # Assert
        {
            "multiplicity": lambda: np.testing.assert_array_equal(
                particulator.attributes["multiplicity"].to_ndarray(),
                np.array(params["n_expected"]),
            ),
            "v": lambda: np.testing.assert_array_almost_equal(
                particulator.attributes["volume"].to_ndarray(),
                np.array(params["v_expected"]),
                decimal=6,
            ),
            "conserve": lambda: np.testing.assert_almost_equal(
                np.sum(
                    particulator.attributes["multiplicity"].to_ndarray()
                    * particulator.attributes["volume"].to_ndarray()
                ),
                np.sum(np.array(params["n_init"]) * np.array(params["v_init"])),
                decimal=6,
            ),
            "deficit": lambda: np.testing.assert_equal(
                breakup_rate_deficit.to_ndarray(), np.array(params["expected_deficit"])
            ),
        }[flag]()

    @staticmethod
    def test_nonnegative_even_if_overflow(
        backend=CPU(),
    ):  # pylint: disable=too-many-locals
        n_sd = 2**5

        dv = 1 * si.m**3
        dt = 1 * si.s
        env = Box(dv=dv, dt=dt)
        builder = Builder(n_sd=n_sd, backend=backend, environment=env)
        env["rhod"] = 1.0

        norm_factor = 100 / si.cm**3 * si.m**3
        X0 = Trivia.volume(const, radius=30.531 * si.micrometres)
        spectrum = spectra.Exponential(norm_factor=norm_factor, scale=X0)
        attributes = {}
        attributes["volume"], attributes["multiplicity"] = ConstantMultiplicity(
            spectrum
        ).sample(n_sd)

        mu = Trivia.volume(const, radius=100 * si.um)
        fragmentation = breakup_fragmentations.Exponential(scale=mu)
        kernel = Geometric()
        coal_eff = ConstEc(Ec=0.01)
        break_eff = ConstEb(Eb=1.0)
        breakup = Collision(
            collision_kernel=kernel,
            breakup_efficiency=break_eff,
            coalescence_efficiency=coal_eff,
            fragmentation_function=fragmentation,
            warn_overflows=False,
        )
        builder.add_dynamic(breakup)

        radius_bins_edges = np.logspace(
            np.log10(0.01 * si.um), np.log10(5000 * si.um), num=64, endpoint=True
        )
        products = (
            ParticleVolumeVersusRadiusLogarithmSpectrum(
                radius_bins_edges=radius_bins_edges, name="dv/dlnr"
            ),
        )
        particulator = builder.build(attributes, products)

        t_end = 100
        particulator.run(t_end - particulator.n_steps)

        assert (particulator.attributes["multiplicity"].to_ndarray() > 0).all()

    @staticmethod
    @pytest.mark.parametrize(
        "params",
        [
            {
                "gamma": [2.0],
                "n_init": [1, 1],
                "v_init": [1, 1],
                "n_expected": [2, 2],
                "v_expected": [0.5, 0.5],
                "expected_deficit": [0.0],
                "is_first_in_pair": [True, False],
                "frag_volume": [0.5],
            },
            {
                "gamma": [3.0],
                "n_init": [9, 2],
                "v_init": [1, 2],
                "n_expected": [2, 11],
                "v_expected": [1, 1],
                "expected_deficit": [0.0],
                "is_first_in_pair": [True, False],
                "frag_volume": [1.0],
            },
        ],
    )
    @pytest.mark.parametrize("flag", ("multiplicity", "v", "conserve", "deficit"))
    def test_while_loop_multi_breakup(
        params, flag, backend_class=CPU
    ):  # pylint:disable=too-many-locals
        # Arrange
        n_init = params["n_init"]
        n_sd = len(n_init)
        env = Box(dv=np.NaN, dt=np.NaN)
        builder = Builder(
            n_sd, backend_class(Formulae(handle_all_breakups=True)), environment=env
        )
        particulator = builder.build(
            attributes={
                "multiplicity": np.asarray(n_init),
                "volume": np.asarray(params["v_init"]),
            },
            products=(),
        )

        n_pairs = n_sd // 2
        rand = [1.0] * n_pairs
        Eb = [1.0] * n_pairs
        pairwise_zeros = particulator.PairwiseStorage.from_ndarray(
            np.array([0.0] * n_pairs)
        )
        general_zeros = particulator.Storage.from_ndarray(np.array([0] * n_sd))

        gamma = particulator.PairwiseStorage.from_ndarray(np.array(params["gamma"]))
        rand = particulator.PairwiseStorage.from_ndarray(np.array(rand))
        Eb = particulator.PairwiseStorage.from_ndarray(np.array(Eb))
        breakup_rate = particulator.Storage.from_ndarray(np.array([0]))
        breakup_rate_deficit = particulator.Storage.from_ndarray(np.array([0]))

        frag_mass = volume_to_mass(particulator, params["frag_volume"])
        frag_mass = particulator.PairwiseStorage.from_ndarray(
            np.array(frag_mass, dtype=float)
        )
        is_first_in_pair = particulator.PairIndicator(n_sd)
        is_first_in_pair.indicator[:] = particulator.Storage.from_ndarray(
            np.asarray(params["is_first_in_pair"])
        )

        # Act
        particulator.collision_coalescence_breakup(
            enable_breakup=True,
            gamma=gamma,
            rand=rand,
            Ec=pairwise_zeros,
            Eb=Eb,
            fragment_mass=frag_mass,
            coalescence_rate=general_zeros,
            breakup_rate=breakup_rate,
            breakup_rate_deficit=breakup_rate_deficit,
            is_first_in_pair=is_first_in_pair,
            warn_overflows=True,
            max_multiplicity=DEFAULTS.max_multiplicity,
        )

        # Assert
        {
            "multiplicity": lambda: np.testing.assert_array_equal(
                particulator.attributes["multiplicity"].to_ndarray(),
                np.array(params["n_expected"]),
            ),
            "v": lambda: np.testing.assert_array_almost_equal(
                particulator.attributes["volume"].to_ndarray(),
                np.array(params["v_expected"]),
                decimal=6,
            ),
            "conserve": lambda: np.testing.assert_almost_equal(
                np.sum(
                    particulator.attributes["multiplicity"].to_ndarray()
                    * particulator.attributes["volume"].to_ndarray()
                ),
                np.sum(np.array(params["n_init"]) * np.array(params["v_init"])),
                decimal=6,
            ),
            "deficit": lambda: np.testing.assert_equal(
                breakup_rate_deficit.to_ndarray(), np.array(params["expected_deficit"])
            ),
        }[flag]()


def get_smaller_of_pairs(is_first_in_pair, n_init):
    return np.where(
        np.roll(is_first_in_pair.indicator, shift=1), np.asarray(n_init), 0.0
    )
