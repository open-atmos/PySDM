"""CPU implementation of backend methods for particle spawning"""

from functools import cached_property

import numba

from PySDM.backends.impl_common.backend_methods import BackendMethods


class SpawningMethods(BackendMethods):  # pylint: disable=too-few-public-methods
    @cached_property
    def _spawning(self):
        @numba.njit(**{**self.default_jit_flags, "parallel": False})
        def body(  # pylint: disable=too-many-arguments
            idx,
            multiplicity,
            extensive_attributes,
            spawned_particle_index,
            spawned_particle_multiplicity,
            spawned_particle_extensive_attributes,
            number_of_super_particles_to_spawn: int,
        ):
            number_of_super_particles_already_injected = 0
            # TODO #1387 start enumerating from the end of valid particle set
            for i, mult in enumerate(multiplicity):
                if (
                    number_of_super_particles_to_spawn
                    == number_of_super_particles_already_injected
                ):
                    break
                if mult == 0:
                    idx[i] = -1
                    s = spawned_particle_index[
                        number_of_super_particles_already_injected
                    ]
                    number_of_super_particles_already_injected += 1
                    multiplicity[i] = spawned_particle_multiplicity[s]
                    for a in range(len(extensive_attributes)):
                        extensive_attributes[a, i] = (
                            spawned_particle_extensive_attributes[a, s]
                        )
            assert (
                number_of_super_particles_to_spawn
                == number_of_super_particles_already_injected
            )

        return body

    def spawning(
        self,
        *,
        idx,
        multiplicity,
        extensive_attributes,
        spawned_particle_index,
        spawned_particle_multiplicity,
        spawned_particle_extensive_attributes,
        number_of_super_particles_to_spawn: int,
    ):
        self._spawning(
            idx=idx.data,
            multiplicity=multiplicity.data,
            extensive_attributes=extensive_attributes.data,
            spawned_particle_index=spawned_particle_index.data,
            spawned_particle_multiplicity=spawned_particle_multiplicity.data,
            spawned_particle_extensive_attributes=spawned_particle_extensive_attributes.data,
            number_of_super_particles_to_spawn=number_of_super_particles_to_spawn,
        )
