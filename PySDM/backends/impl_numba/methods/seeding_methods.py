from functools import cached_property

import numba

from PySDM.backends.impl_common.backend_methods import BackendMethods


class SeedingMethods(BackendMethods):
    @cached_property
    def _seeding(self):
        @numba.njit(**{**self.default_jit_flags, "parallel": False})
        def body(
            idx,
            multiplicity,
            extensive_attributes,
            seeded_particle_multiplicity,
            seeded_particle_extensive_attributes,
        ):
            for i, mult in enumerate(multiplicity):
                if mult == 0:
                    idx[i] = -1
                    multiplicity[i] = seeded_particle_multiplicity
                    for a in range(len(extensive_attributes)):
                        extensive_attributes[a, i] = (
                            seeded_particle_extensive_attributes[a]
                        )
                    break

        return body

    def seeding(
        self,
        idx,
        multiplicity,
        extensive_attributes,
        seeded_particle_multiplicity,
        seeded_particle_extensive_attributes,
    ):
        self._seeding(
            idx=idx.data,
            multiplicity=multiplicity.data,
            extensive_attributes=extensive_attributes.data,
            seeded_particle_multiplicity=seeded_particle_multiplicity,
            seeded_particle_extensive_attributes=tuple(
                seeded_particle_extensive_attributes.values()
            ),
        )
