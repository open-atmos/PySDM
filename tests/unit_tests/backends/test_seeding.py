""" Seeding backend tests of injection logic """

import numpy as np
import pytest

from PySDM import Builder
from PySDM.environments import Box
from PySDM.backends import CPU
from PySDM.physics import si


class TestSeeding:
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
    def test_number_of_super_particles_to_inject(
        n_sd,
        number_of_super_particles_to_inject,
        dt=1,
        dv=1,
    ):
        """unit test for injection logic on: number_of_super_particles_to_inject"""
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
    def test_seeded_particle_index_multiplicity_extensive_attributes(
        seeded_particle_index,
        n_sd=3,
        number_of_super_particles_to_inject=3,
        dt=1,
        dv=1,
    ):
        """unit test for injection logic on: seeded_particle_index, \
            seeded_particle_multiplicity, seeded_particle_extensive_attributes"""

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
