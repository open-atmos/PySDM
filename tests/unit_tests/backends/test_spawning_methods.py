"""Seeding backend tests of spawning logic"""

from contextlib import nullcontext

import numpy as np
import pytest

from PySDM import Builder
from PySDM.environments import Box
from PySDM.backends import CPU
from PySDM.physics import si


class TestSpawningMethods:
    max_number_to_spawn = 4

    @staticmethod
    @pytest.mark.parametrize(
        "n_sd, number_of_super_particles_to_spawn, context",
        (
            (1, 1, nullcontext()),
            (
                1,
                2,
                pytest.raises(
                    ValueError, match="spawn more super particles than space available"
                ),
            ),
            (max_number_to_spawn, max_number_to_spawn - 1, nullcontext()),
            (max_number_to_spawn, max_number_to_spawn, nullcontext()),
            (
                max_number_to_spawn + 2,
                max_number_to_spawn + 1,
                pytest.raises(
                    ValueError,
                    match="spawn multiple super particles with the same attributes",
                ),
            ),
        ),
    )
    def test_number_of_super_particles_to_spawn(
        n_sd,
        number_of_super_particles_to_spawn,
        context,
        dt=1,
        dv=1,
    ):
        # arrange
        builder = Builder(n_sd, CPU(), Box(dt, dv))
        particulator = builder.build(
            attributes={
                "multiplicity": np.full(n_sd, np.nan),
                "water mass": np.zeros(n_sd),
            },
        )

        spawned_particle_extensive_attributes = {
            "water mass": [0.0001 * si.ng] * TestSpawningMethods.max_number_to_spawn,
        }
        spawned_particle_multiplicity = [1] * TestSpawningMethods.max_number_to_spawn

        spawned_particle_index = particulator.Index.identity_index(
            len(spawned_particle_multiplicity)
        )
        spawned_particle_multiplicity = particulator.IndexedStorage.from_ndarray(
            spawned_particle_index,
            np.asarray(spawned_particle_multiplicity),
        )
        spawned_particle_extensive_attributes = (
            particulator.IndexedStorage.from_ndarray(
                spawned_particle_index,
                np.asarray(list(spawned_particle_extensive_attributes.values())),
            )
        )

        # act
        with context:
            particulator.spawning(
                spawned_particle_index=spawned_particle_index,
                spawned_particle_multiplicity=spawned_particle_multiplicity,
                spawned_particle_extensive_attributes=spawned_particle_extensive_attributes,
                number_of_super_particles_to_spawn=number_of_super_particles_to_spawn,
            )

            # assert
            assert (
                number_of_super_particles_to_spawn
                == particulator.attributes.super_droplet_count
            )

    @staticmethod
    @pytest.mark.parametrize(
        "spawned_particle_index, context",
        (
            ([0, 0, 0], nullcontext()),
            ([0, 1, 2], nullcontext()),
            ([2, 1, 0], nullcontext()),
            (
                [0],
                pytest.raises(
                    ValueError,
                    match=" multiple super particles with the same attributes",
                ),
            ),
        ),
    )
    def test_spawned_particle_index_multiplicity_extensive_attributes(
        spawned_particle_index,
        context,
        n_sd=3,
        number_of_super_particles_to_spawn=3,
    ):
        # arrange
        builder = Builder(n_sd, CPU(), Box(dt=np.nan, dv=np.nan))
        particulator = builder.build(
            attributes={
                "multiplicity": np.full(n_sd, np.nan),
                "water mass": np.zeros(n_sd),
            },
        )

        spawned_particle_extensive_attributes = {
            "water mass": [0.0001, 0.0003, 0.0002],
        }
        spawned_particle_multiplicity = [1, 2, 3]

        spawned_particle_index_impl = particulator.Index.from_ndarray(
            np.asarray(spawned_particle_index)
        )
        spawned_particle_multiplicity_impl = particulator.IndexedStorage.from_ndarray(
            spawned_particle_index_impl,
            np.asarray(spawned_particle_multiplicity),
        )
        spawned_particle_extensive_attributes_impl = (
            particulator.IndexedStorage.from_ndarray(
                spawned_particle_index_impl,
                np.asarray(list(spawned_particle_extensive_attributes.values())),
            )
        )

        # act
        with context:
            particulator.spawning(
                spawned_particle_index=spawned_particle_index_impl,
                spawned_particle_multiplicity=spawned_particle_multiplicity_impl,
                spawned_particle_extensive_attributes=spawned_particle_extensive_attributes_impl,
                number_of_super_particles_to_spawn=number_of_super_particles_to_spawn,
            )

            # assert
            np.testing.assert_array_equal(
                particulator.attributes["multiplicity"].to_ndarray(),
                np.asarray(spawned_particle_multiplicity)[spawned_particle_index],
            )
            np.testing.assert_array_equal(
                particulator.attributes["water mass"].to_ndarray(),
                np.asarray(spawned_particle_extensive_attributes["water mass"])[
                    spawned_particle_index
                ],
            )
