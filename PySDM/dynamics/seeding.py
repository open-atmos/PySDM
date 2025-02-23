"""particle injection handling, requires initalising a simulation with
enough particles flagged with NaN multiplicity (translated to zeros
at multiplicity discretisation"""

from collections.abc import Sized

import numpy as np

from PySDM.dynamics.impl import register_dynamic
from PySDM.initialisation import discretise_multiplicities


@register_dynamic()
class Seeding:
    def __init__(
        self,
        *,
        super_droplet_injection_rate: callable,
        seeded_particle_extensive_attributes: dict,
        seeded_particle_multiplicity: Sized,
    ):
        for attr in seeded_particle_extensive_attributes.values():
            assert len(seeded_particle_multiplicity) == len(attr)
        self.particulator = None
        self.super_droplet_injection_rate = super_droplet_injection_rate
        self.seeded_particle_extensive_attributes = seeded_particle_extensive_attributes
        self.seeded_particle_multiplicity = seeded_particle_multiplicity
        self.rnd = None
        self.u01 = None
        self.index = None

    def register(self, builder):
        self.particulator = builder.particulator

    def post_register_setup_when_attributes_are_known(self):
        if tuple(self.particulator.attributes.get_extensive_attribute_keys()) != tuple(
            self.seeded_particle_extensive_attributes.keys()
        ):
            raise ValueError(
                f"extensive attributes ({self.seeded_particle_extensive_attributes.keys()})"
                " do not match those used in particulator"
                f" ({self.particulator.attributes.get_extensive_attribute_keys()})"
            )

        self.index = self.particulator.Index.identity_index(
            len(self.seeded_particle_multiplicity)
        )
        if len(self.seeded_particle_multiplicity) > 1:
            self.rnd = self.particulator.Random(
                len(self.seeded_particle_multiplicity), self.particulator.formulae.seed
            )
            self.u01 = self.particulator.Storage.empty(
                len(self.seeded_particle_multiplicity), dtype=float
            )
        self.seeded_particle_multiplicity = (
            self.particulator.IndexedStorage.from_ndarray(
                self.index,
                discretise_multiplicities(
                    np.asarray(self.seeded_particle_multiplicity)
                ),
            )
        )
        self.seeded_particle_extensive_attributes = (
            self.particulator.IndexedStorage.from_ndarray(
                self.index,
                np.asarray(list(self.seeded_particle_extensive_attributes.values())),
            )
        )

    def __call__(self):
        if self.particulator.n_steps == 0:
            self.post_register_setup_when_attributes_are_known()

        time = self.particulator.n_steps * self.particulator.dt
        number_of_super_particles_to_inject = self.super_droplet_injection_rate(time)

        if number_of_super_particles_to_inject > 0:
            assert number_of_super_particles_to_inject <= len(
                self.seeded_particle_multiplicity
            )

            if self.rnd is not None:
                self.u01.urand(self.rnd)
                # TODO #1387 make shuffle smarter
                # e.g. don't need to shuffle if only one type of seed particle
                # or if the number of super particles to inject
                # is equal to the number of possible seeds
                self.index.shuffle(self.u01)
            self.particulator.seeding(
                seeded_particle_index=self.index,
                number_of_super_particles_to_inject=number_of_super_particles_to_inject,
                seeded_particle_multiplicity=self.seeded_particle_multiplicity,
                seeded_particle_extensive_attributes=self.seeded_particle_extensive_attributes,
            )
