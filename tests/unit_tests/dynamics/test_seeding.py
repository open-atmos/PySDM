""" Seeding dynamic tests """

from collections import namedtuple

import numpy as np
import pytest

from matplotlib import pyplot

from PySDM import Builder
from PySDM.products import SuperDropletCountPerGridbox, Time
from PySDM.backends import CPU
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
                    val.append(float(particulator.products[key].get()))
            for key in products:
                products[key] = np.array(products[key])
            return products

        # act
        common_seeding_ctor_args = {
            "seeded_particle_multiplicity": [1],
            "seeded_particle_extensive_attributes": {
                "water mass": [0.001 * si.ng],
            },
        }
        output = {
            "zero_injection_rate": simulation(
                dynamics=(
                    Seeding(
                        super_droplet_injection_rate=lambda time: 0,
                        **common_seeding_ctor_args
                    ),
                )
            ),
            "positive_injection_rate": simulation(
                dynamics=(
                    Seeding(
                        super_droplet_injection_rate=lambda time: (
                            1 if 10 * si.min <= time < 15 * si.min else 0
                        ),
                        **common_seeding_ctor_args
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

    max_number_to_inject = 4

    @staticmethod
    @pytest.mark.parametrize(
        "n_sd, number_of_super_particles_to_inject",
        (
            (1, 1),
            pytest.param(1, 2, marks=pytest.mark.xfail(strict=True)),
            (max_number_to_inject, max_number_to_inject - 1),
            (max_number_to_inject, max_number_to_inject),
            pytest.param(
                max_number_to_inject + 2,
                max_number_to_inject + 1,
                marks=pytest.mark.xfail(strict=True),
            ),
        ),
    )
    def test_seeding_number_of_super_particles_to_inject(
        n_sd,
        number_of_super_particles_to_inject,
        dt=1,
        dv=1,
    ):
        """unit test for injection logic on: number_of_super_particles_to_inject

        FUTURE TESTS:
            seeded_particle_multiplicity,
            seeded_particle_extensive_attributes,
        """
        # arrange
        builder = Builder(n_sd, CPU(), Box(dt, dv))
        particulator = builder.build(
            attributes={
                "multiplicity": np.full(n_sd, np.nan),
                "water mass": np.zeros(n_sd),
            },
        )

        seeded_particle_extensive_attributes = {
            "water mass": [0.0001 * si.ng] * TestSeeding.max_number_to_inject,
        }
        seeded_particle_multiplicity = [1] * TestSeeding.max_number_to_inject

        seeded_particle_index = particulator.Index.identity_index(
            len(seeded_particle_multiplicity)
        )
        seeded_particle_multiplicity = particulator.IndexedStorage.from_ndarray(
            seeded_particle_index,
            np.asarray(seeded_particle_multiplicity),
        )
        seeded_particle_extensive_attributes = particulator.IndexedStorage.from_ndarray(
            seeded_particle_index,
            np.asarray(list(seeded_particle_extensive_attributes.values())),
        )

        # act
        particulator.seeding(
            seeded_particle_index=seeded_particle_index,
            seeded_particle_multiplicity=seeded_particle_multiplicity,
            seeded_particle_extensive_attributes=seeded_particle_extensive_attributes,
            number_of_super_particles_to_inject=number_of_super_particles_to_inject,
        )

        # assert
        assert (
            number_of_super_particles_to_inject
            == particulator.attributes.super_droplet_count
        )

    @staticmethod
    @pytest.mark.parametrize(
        "seeded_particle_index",
        (
            [0, 0, 0],
            [0, 1, 2],
            [2, 1, 0],
            pytest.param([0], marks=pytest.mark.xfail(strict=True)),
        ),
    )
    def test_seeded_particle_index(
        seeded_particle_index,
        n_sd=3,
        number_of_super_particles_to_inject=3,
        dt=1,
        dv=1,
    ):
        """unit test for injection logic on: seeded_particle_index"""

        # arrange
        builder = Builder(n_sd, CPU(), Box(dt, dv))
        particulator = builder.build(
            attributes={
                "multiplicity": np.full(n_sd, np.nan),
                "water mass": np.zeros(n_sd),
            },
        )

        seeded_particle_extensive_attributes = {
            "water mass": [0.0001, 0.0003, 0.0002],
        }
        seeded_particle_multiplicity = [1, 2, 3]

        seeded_particle_index_impl = particulator.Index.from_ndarray(
            np.asarray(seeded_particle_index)
        )
        seeded_particle_multiplicity_impl = particulator.IndexedStorage.from_ndarray(
            seeded_particle_index_impl,
            np.asarray(seeded_particle_multiplicity),
        )
        seeded_particle_extensive_attributes_impl = (
            particulator.IndexedStorage.from_ndarray(
                seeded_particle_index_impl,
                np.asarray(list(seeded_particle_extensive_attributes.values())),
            )
        )

        # act
        particulator.seeding(
            seeded_particle_index=seeded_particle_index_impl,
            seeded_particle_multiplicity=seeded_particle_multiplicity_impl,
            seeded_particle_extensive_attributes=seeded_particle_extensive_attributes_impl,
            number_of_super_particles_to_inject=number_of_super_particles_to_inject,
        )

        # assert
        np.testing.assert_array_equal(
            particulator.attributes["multiplicity"].to_ndarray(),
            np.asarray(seeded_particle_multiplicity)[seeded_particle_index],
        )
        np.testing.assert_array_equal(
            particulator.attributes["water mass"].to_ndarray(),
            np.asarray(seeded_particle_extensive_attributes["water mass"])[
                seeded_particle_index
            ],
        )
