""" CPU implementation of backend methods for particle injections """

from functools import cached_property

import numba

from PySDM.backends.impl_common.backend_methods import BackendMethods


class SeedingMethods(BackendMethods):  # pylint: disable=too-few-public-methods
    @cached_property
    def _seeding(self):
        @numba.njit(**{**self.default_jit_flags, "parallel": False})
        def body(  # pylint: disable=too-many-arguments
            idx,
            multiplicity,
            extensive_attributes,
            seeded_particle_multiplicity,
            seeded_particle_extensive_attributes,
            number_of_super_particles_to_inject: int,
        ):
            # TODO #1367 start enumerating from the end of valid particle set
            for i, mult in enumerate(multiplicity):
                if number_of_super_particles_to_inject == 0:
                    break
                if mult == 0:
                    idx[i] = -1
                    multiplicity[i] = seeded_particle_multiplicity
                    for a in range(len(extensive_attributes)):
                        extensive_attributes[a, i] = (
                            seeded_particle_extensive_attributes[a]
                        )
                    number_of_super_particles_to_inject -= 1
            assert number_of_super_particles_to_inject == 0

        return body

    def seeding(
        self,
        *,
        idx,
        multiplicity,
        extensive_attributes,
        seeded_particle_multiplicity,
        seeded_particle_extensive_attributes,
        number_of_super_particles_to_inject: int,
    ):
        self._seeding(
            idx=idx.data,
            multiplicity=multiplicity.data,
            extensive_attributes=extensive_attributes.data,
            seeded_particle_multiplicity=seeded_particle_multiplicity,
            seeded_particle_extensive_attributes=tuple(
                seeded_particle_extensive_attributes.values()
            ),
            number_of_super_particles_to_inject=number_of_super_particles_to_inject,
        )
