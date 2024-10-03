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
            cell_id,
            cell_origin,
            pos_cell,
            volume,
            extensive_attributes,
            seeded_particle_index,
            seeded_particle_multiplicity,
            seeded_particle_cell_id,
            seeded_particle_cell_origin,
            seeded_particle_pos_cell,
            seeded_particle_extensive_attributes,
            seeded_particle_volume,
            number_of_super_particles_to_inject: int,
        ):
            number_of_super_particles_already_injected = 0
            # TODO #1387 start enumerating from the end of valid particle set
            for i, mult in enumerate(multiplicity):
                if (
                    number_of_super_particles_to_inject
                    == number_of_super_particles_already_injected
                ):
                    break
                if mult == 0:
                    idx[i] = -1
                    s = seeded_particle_index[
                        number_of_super_particles_already_injected
                    ]
                    number_of_super_particles_already_injected += 1
                    multiplicity[i] = seeded_particle_multiplicity[s]
                    cell_id[i] = seeded_particle_cell_id[s]
                    cell_origin[0][i] = seeded_particle_cell_origin[0][s]
                    pos_cell[0][i] = seeded_particle_pos_cell[0][s]
                    for a in range(len(extensive_attributes)):
                        extensive_attributes[a, i] = (
                            seeded_particle_extensive_attributes[a, s]
                        )
                    volume[i] = seeded_particle_volume[s]
            assert (
                number_of_super_particles_to_inject
                == number_of_super_particles_already_injected
            )

        return body

    def seeding(
        self,
        *,
        idx,
        multiplicity,
        cell_id,
        cell_origin,
        pos_cell,
        volume,
        extensive_attributes,
        seeded_particle_index,
        seeded_particle_multiplicity,
        seeded_particle_cell_id,
        seeded_particle_cell_origin,
        seeded_particle_pos_cell,
        seeded_particle_extensive_attributes,
        seeded_particle_volume,
        number_of_super_particles_to_inject: int,
    ):
        self._seeding(
            idx=idx.data,
            multiplicity=multiplicity.data,
            cell_id=cell_id.data,
            cell_origin=cell_origin.data,
            pos_cell=pos_cell.data,
            volume=volume.data,
            extensive_attributes=extensive_attributes.data,
            seeded_particle_index=seeded_particle_index.data,
            seeded_particle_multiplicity=seeded_particle_multiplicity.data,
            seeded_particle_cell_id=seeded_particle_cell_id.data,
            seeded_particle_cell_origin=seeded_particle_cell_origin.data,
            seeded_particle_pos_cell=seeded_particle_pos_cell.data,
            seeded_particle_extensive_attributes=seeded_particle_extensive_attributes.data,
            seeded_particle_volume=seeded_particle_volume.data,
            number_of_super_particles_to_inject=number_of_super_particles_to_inject,
        )
