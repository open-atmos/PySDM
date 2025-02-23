"""Seeding dynamic tests"""

from collections import namedtuple

import numpy as np
import pytest

from matplotlib import pyplot

from PySDM import Builder
from PySDM.products import SuperDropletCountPerGridbox, Time
from PySDM.backends import CPU
from PySDM.backends.impl_common.index import make_Index
from PySDM.backends.impl_common.indexed_storage import make_IndexedStorage
from PySDM.dynamics import Seeding
from PySDM.environments import Box
from PySDM.physics import si


class TestSeeding:
    @staticmethod
    def test_zero_injection_rate_same_as_no_seeding_monodisperse(
        plot=False, backend_instance=CPU()
    ):
        """a not-so-unit test checking that results of a box simulation
        are the same without seeding as with a zero injection rate"""
        # arrange
        n_sd_seeding = 100
        n_sd_initial = 100

        def simulation(*, dynamics):
            t_max = 20 * si.min
            timestep = 15 * si.s
            dv = 1 * si.cm**3

            builder = Builder(
                backend=backend_instance,
                n_sd=n_sd_seeding + n_sd_initial,
                environment=Box(dt=timestep, dv=dv),
            )
            for dynamic in dynamics:
                builder.add_dynamic(dynamic)

            particulator = builder.build(
                attributes={
                    k: np.pad(
                        array=v,
                        pad_width=(0, n_sd_seeding),
                        mode="constant",
                        constant_values=np.nan if k == "multiplicity" else 0,
                    )
                    for k, v in {
                        "volume": np.ones(n_sd_initial) * si.um**3,
                        "multiplicity": np.ones(n_sd_initial),
                    }.items()
                },
                products=(
                    SuperDropletCountPerGridbox(name="sd_count"),
                    Time(),
                ),
            )
            products = {"sd_count": [], "time": []}
            for step in range(int(t_max // timestep) + 1):
                if step != 0:
                    particulator.run(steps=1)
                for key, val in products.items():
                    value = particulator.products[key].get()
                    if not isinstance(value, float):
                        (value,) = value
                    val.append(float(value))
            for key in products:
                products[key] = np.array(products[key])
            return products

        # act
        common_seeding_ctor_args = {
            "seeded_particle_multiplicity": [1],
            "seeded_particle_extensive_attributes": {
                "signed water mass": [0.001 * si.ng],
            },
        }
        output = {
            "zero_injection_rate": simulation(
                dynamics=(
                    Seeding(
                        super_droplet_injection_rate=lambda time: 0,
                        **common_seeding_ctor_args,
                    ),
                )
            ),
            "positive_injection_rate": simulation(
                dynamics=(
                    Seeding(
                        super_droplet_injection_rate=lambda time: (
                            1 if 10 * si.min <= time < 15 * si.min else 0
                        ),
                        **common_seeding_ctor_args,
                    ),
                )
            ),
            "no_seeding": simulation(dynamics=()),
        }

        # plot
        pyplot.plot(
            output["zero_injection_rate"]["sd_count"],
            output["zero_injection_rate"]["time"],
            lw=2,
            label="with seeding dynamic, zero injection rate",
        )
        pyplot.plot(
            output["positive_injection_rate"]["sd_count"],
            output["positive_injection_rate"]["time"],
            lw=2,
            ls="--",
            label="with seeding dynamic, positive injection rate",
        )
        pyplot.plot(
            output["no_seeding"]["sd_count"],
            output["no_seeding"]["time"],
            lw=2,
            ls=":",
            label="without seeding dynamic",
        )
        pyplot.xlabel("sd_count")
        pyplot.ylabel("time")
        pyplot.grid()
        pyplot.legend()
        if plot:
            pyplot.show()
        else:
            pyplot.clf()

        # assert
        np.testing.assert_array_equal(
            output["zero_injection_rate"]["sd_count"], output["no_seeding"]["sd_count"]
        )
        np.testing.assert_array_equal(np.diff(output["no_seeding"]["sd_count"]), 0)
        assert np.amax(np.diff(output["positive_injection_rate"]["sd_count"])) >= 0
        assert output["positive_injection_rate"]["sd_count"][0] == n_sd_initial
        assert (
            n_sd_initial
            < output["positive_injection_rate"]["sd_count"][-1]
            < n_sd_initial + n_sd_seeding
        )

    @staticmethod
    def test_attribute_set_match():
        # arrange
        extensive_attributes = ["a", "b", "c"]
        seeding_attributes = ["d", "e", "f"]

        builder = namedtuple(typename="MockBuilder", field_names=("particulator",))(
            particulator=namedtuple(
                typename="MockParticulator", field_names=("n_steps", "attributes")
            )(
                n_steps=0,
                attributes=namedtuple(
                    typename="MockAttributes",
                    field_names=("get_extensive_attribute_keys",),
                )(get_extensive_attribute_keys=lambda: extensive_attributes),
            )
        )

        dynamic = Seeding(
            super_droplet_injection_rate=lambda time: 0,
            seeded_particle_extensive_attributes={
                k: [np.nan] for k in seeding_attributes
            },
            seeded_particle_multiplicity=[0],
        )
        dynamic.register(builder)

        # act
        with pytest.raises(ValueError) as excinfo:
            dynamic()

        # assert
        assert "do not match those used in particulator" in str(excinfo)

    @staticmethod
    @pytest.mark.parametrize(
        "super_droplet_injection_rate, reservoir_length",
        (
            (0, 0),  # shuffle not called
            (0, 2),
            (1, 10),
            (2, 10),
        ),
    )
    def test_seeded_particle_shuffle(
        super_droplet_injection_rate, reservoir_length, backend=CPU()
    ):
        # arrange
        extensive_attributes = ["a"]
        seeding_attributes = ["a"]

        class MockParticulator:  # pylint: disable=too-few-public-methods
            def __init__(self):
                self.seeding_call_count = 0
                self.indices = []

            def seeding(
                self,
                *,
                seeded_particle_index,
                number_of_super_particles_to_inject,  # pylint: disable=unused-argument
                seeded_particle_multiplicity,  # pylint: disable=unused-argument
                seeded_particle_extensive_attributes,  # pylint: disable=unused-argument
            ):
                self.seeding_call_count += 1
                self.indices.append(seeded_particle_index.to_ndarray())

            Index = make_Index(backend)
            IndexedStorage = make_IndexedStorage(backend)
            Random = None if reservoir_length == 0 else backend.Random
            formulae = None if reservoir_length == 0 else backend.formulae
            Storage = None if reservoir_length == 0 else backend.Storage
            n_steps = 0
            dt = np.nan
            attributes = namedtuple(
                typename="MockAttributes",
                field_names=("get_extensive_attribute_keys",),
            )(get_extensive_attribute_keys=lambda: extensive_attributes)

        builder = namedtuple(typename="MockBuilder", field_names=("particulator",))(
            particulator=MockParticulator()
        )

        dynamic = Seeding(
            super_droplet_injection_rate=lambda t: super_droplet_injection_rate,
            seeded_particle_extensive_attributes={
                k: [np.nan] * reservoir_length for k in seeding_attributes
            },
            seeded_particle_multiplicity=[1] * reservoir_length,
        )
        dynamic.register(builder)

        # act
        dynamic()
        dynamic.particulator.n_steps += 1
        dynamic()

        # assert
        assert dynamic.particulator.seeding_call_count == (
            2 if super_droplet_injection_rate > 0 else 0
        )
        if super_droplet_injection_rate > 0:
            assert (
                dynamic.particulator.indices[0] != dynamic.particulator.indices[1]
            ).any()
            assert sorted(dynamic.particulator.indices[0]) == sorted(
                dynamic.particulator.indices[1]
            )
