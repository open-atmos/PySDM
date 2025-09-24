"""
CPU implementation of backend methods for particle displacement (advection and sedimentation)
"""

from functools import cached_property

import numba

from PySDM.backends.impl_numba import conf

from ...impl_common.backend_methods import BackendMethods


@numba.njit(**{**conf.JIT_FLAGS, **{"parallel": False}})
# pylint: disable=too-many-arguments
def calculate_displacement_body_common(
    dim, droplet, scheme, _l, _r, displacement, courant, position_in_cell, cell_id, n_substeps, enable_monte_carlo, u01
):
    displacement[dim, droplet] = scheme(
        position_in_cell[dim, droplet],
        cell_id[droplet],
        courant[_l] / n_substeps,
        courant[_r] / n_substeps,
        enable_monte_carlo,
        u01
    )


class DisplacementMethods(BackendMethods):
    @staticmethod
    @numba.njit(**{**conf.JIT_FLAGS, **{"parallel": False, "cache": False}})
    # pylint: disable=too-many-arguments
    def calculate_displacement_body_1d(
        dim, scheme, displacement, courant, cell_origin, position_in_cell, cell_id, n_substeps, enable_monte_carlo, rng
    ):
        length = displacement.shape[1]
        for droplet in numba.prange(length):  # pylint: disable=not-an-iterable
            # Arakawa-C grid
            _l = cell_origin[0, droplet]
            _r = cell_origin[0, droplet] + 1
            u01 = rng[droplet]
            calculate_displacement_body_common(
                dim,
                droplet,
                scheme,
                _l,
                _r,
                displacement,
                courant,
                position_in_cell,
                cell_id,
                n_substeps,
                enable_monte_carlo,
                u01,
            )

    @staticmethod
    @numba.njit(**{**conf.JIT_FLAGS, **{"parallel": False, "cache": False}})
    # pylint: disable=too-many-arguments
    def calculate_displacement_body_2d(
        dim, scheme, displacement, courant, cell_origin, position_in_cell, cell_id, n_substeps, enable_monte_carlo, rng
    ):
        length = displacement.shape[1]
        for droplet in numba.prange(length):  # pylint: disable=not-an-iterable
            # Arakawa-C grid
            _l = (
                cell_origin[0, droplet],
                cell_origin[1, droplet],
            )
            _r = (
                cell_origin[0, droplet] + 1 * (dim == 0),
                cell_origin[1, droplet] + 1 * (dim == 1),
            )
            u01 = rng[droplet]
            calculate_displacement_body_common(
                dim,
                droplet,
                scheme,
                _l,
                _r,
                displacement,
                courant,
                position_in_cell,
                cell_id,
                n_substeps,
                enable_monte_carlo,
                u01,
            )

    @staticmethod
    @numba.njit(**{**conf.JIT_FLAGS, **{"parallel": False, "cache": False}})
    # pylint: disable=too-many-arguments
    def calculate_displacement_body_3d(
        dim, scheme, displacement, courant, cell_origin, position_in_cell, cell_id, n_substeps, enable_monte_carlo, rng
    ):
        n_sd = displacement.shape[1]
        for droplet in numba.prange(n_sd):  # pylint: disable=not-an-iterable
            # Arakawa-C grid
            _l = (
                cell_origin[0, droplet],
                cell_origin[1, droplet],
                cell_origin[2, droplet],
            )
            _r = (
                cell_origin[0, droplet] + 1 * (dim == 0),
                cell_origin[1, droplet] + 1 * (dim == 1),
                cell_origin[2, droplet] + 1 * (dim == 2),
            )
            u01 = rng[droplet]
            calculate_displacement_body_common(
                dim,
                droplet,
                scheme,
                _l,
                _r,
                displacement,
                courant,
                position_in_cell,
                cell_id,
                n_substeps,
                enable_monte_carlo,
                u01,
            )

    def calculate_displacement(
        self, *, dim, displacement, courant, cell_origin, position_in_cell, cell_id, n_substeps, enable_monte_carlo, rng
    ):
        n_dims = len(courant.shape)
        scheme = self.formulae.particle_advection.displacement
        if n_dims == 1:
            DisplacementMethods.calculate_displacement_body_1d(
                dim,
                scheme,
                displacement.data,
                courant.data,
                cell_origin.data,
                position_in_cell.data,
                cell_id.data,
                n_substeps,
                enable_monte_carlo,
                rng.uniform(0., 1., displacement.data.shape[1]),
            )
        elif n_dims == 2:
            DisplacementMethods.calculate_displacement_body_2d(
                dim,
                scheme,
                displacement.data,
                courant.data,
                cell_origin.data,
                position_in_cell.data,
                cell_id.data,
                n_substeps,
                enable_monte_carlo,
                rng.uniform(0., 1., displacement.data.shape[1]),
            )
        elif n_dims == 3:
            DisplacementMethods.calculate_displacement_body_3d(
                dim,
                scheme,
                displacement.data,
                courant.data,
                cell_origin.data,
                position_in_cell.data,
                cell_id.data,
                n_substeps,
                enable_monte_carlo,
                rng.uniform(0., 1., displacement.data.shape[1]),
            )
        else:
            raise NotImplementedError()

    @cached_property
    def _flag_precipitated_body(self):
        @numba.njit(**{**self.default_jit_flags, "parallel": False})
        # pylint: disable=too-many-arguments
        def body(
            cell_origin,
            position_in_cell,
            water_mass,
            multiplicity,
            idx,
            length,
            healthy,
            precipitation_counting_level_index,
            displacement,
        ):
            rainfall_mass = 0.0
            flag = len(idx)
            for i in range(length):
                position_within_column = (
                    cell_origin[-1, idx[i]] + position_in_cell[-1, idx[i]]
                )
                if (
                    # falling
                    displacement[-1, idx[i]] < 0
                    and
                    # and crossed precip-counting level
                    position_within_column < precipitation_counting_level_index
                ):
                    rainfall_mass += abs(water_mass[idx[i]]) * \
                        multiplicity[idx[i]]
                    idx[i] = flag
                    healthy[0] = 0
            return rainfall_mass

        return body

    @cached_property
    def _flag_out_of_column_body(self):
        @numba.njit(**{**self.default_jit_flags, "parallel": False})
        # pylint: disable=too-many-arguments
        def body(
            cell_origin, position_in_cell, idx, length, healthy, domain_top_level_index
        ):
            flag = len(idx)
            for i in range(length):
                position_within_column = (
                    cell_origin[-1, idx[i]] + position_in_cell[-1, idx[i]]
                )
                if (
                    position_within_column < 0
                    or position_within_column > domain_top_level_index
                ):
                    idx[i] = flag
                    healthy[0] = 0

        return body

    # pylint: disable=too-many-arguments
    def flag_precipitated(
        self,
        *,
        cell_origin,
        position_in_cell,
        water_mass,
        multiplicity,
        idx,
        length,
        healthy,
        precipitation_counting_level_index,
        displacement,
    ) -> float:
        """return a scalar value corresponding to the mass of water (all phases) that crossed
        the bottom boundary of the entire domain"""
        return self._flag_precipitated_body(
            cell_origin.data,
            position_in_cell.data,
            water_mass.data,
            multiplicity.data,
            idx.data,
            length,
            healthy.data,
            precipitation_counting_level_index,
            displacement.data,
        )

    # pylint: disable=too-many-arguments
    def flag_out_of_column(
        self,
        cell_origin,
        position_in_cell,
        idx,
        length,
        healthy,
        domain_top_level_index,
    ):
        self._flag_out_of_column_body(
            cell_origin.data,
            position_in_cell.data,
            idx.data,
            length,
            healthy.data,
            domain_top_level_index,
        )
