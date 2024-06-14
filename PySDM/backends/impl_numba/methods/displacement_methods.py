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
    dim, droplet, scheme, _l, _r, displacement, courant, position_in_cell, n_substeps
):
    displacement[dim, droplet] = scheme(
        position_in_cell[dim, droplet],
        courant[_l] / n_substeps,
        courant[_r] / n_substeps,
    )


class DisplacementMethods(BackendMethods):
    @staticmethod
    @numba.njit(**{**conf.JIT_FLAGS, **{"parallel": False, "cache": False}})
    # pylint: disable=too-many-arguments
    def calculate_displacement_body_1d(
        dim, scheme, displacement, courant, cell_origin, position_in_cell, n_substeps
    ):
        length = displacement.shape[1]
        for droplet in numba.prange(length):  # pylint: disable=not-an-iterable
            # Arakawa-C grid
            _l = cell_origin[0, droplet]
            _r = cell_origin[0, droplet] + 1
            calculate_displacement_body_common(
                dim,
                droplet,
                scheme,
                _l,
                _r,
                displacement,
                courant,
                position_in_cell,
                n_substeps,
            )

    @staticmethod
    @numba.njit(**{**conf.JIT_FLAGS, **{"parallel": False, "cache": False}})
    # pylint: disable=too-many-arguments
    def calculate_displacement_body_2d(
        dim, scheme, displacement, courant, cell_origin, position_in_cell, n_substeps
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
            calculate_displacement_body_common(
                dim,
                droplet,
                scheme,
                _l,
                _r,
                displacement,
                courant,
                position_in_cell,
                n_substeps,
            )

    @staticmethod
    @numba.njit(**{**conf.JIT_FLAGS, **{"parallel": False, "cache": False}})
    # pylint: disable=too-many-arguments
    def calculate_displacement_body_3d(
        dim, scheme, displacement, courant, cell_origin, position_in_cell, n_substeps
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
            calculate_displacement_body_common(
                dim,
                droplet,
                scheme,
                _l,
                _r,
                displacement,
                courant,
                position_in_cell,
                n_substeps,
            )

    def calculate_displacement(
        self, *, dim, displacement, courant, cell_origin, position_in_cell, n_substeps
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
                n_substeps,
            )
        elif n_dims == 2:
            DisplacementMethods.calculate_displacement_body_2d(
                dim,
                scheme,
                displacement.data,
                courant.data,
                cell_origin.data,
                position_in_cell.data,
                n_substeps,
            )
        elif n_dims == 3:
            DisplacementMethods.calculate_displacement_body_3d(
                dim,
                scheme,
                displacement.data,
                courant.data,
                cell_origin.data,
                position_in_cell.data,
                n_substeps,
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
            volume,
            multiplicity,
            idx,
            length,
            healthy,
            precipitation_counting_level_index,
            displacement,
        ):
            rainfall = 0.0
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
                    rainfall += volume[idx[i]] * multiplicity[idx[i]]  # TODO #599
                    idx[i] = flag
                    healthy[0] = 0
            return rainfall

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
        cell_origin,
        position_in_cell,
        volume,
        multiplicity,
        idx,
        length,
        healthy,
        precipitation_counting_level_index,
        displacement,
    ) -> float:
        return self._flag_precipitated_body(
            cell_origin.data,
            position_in_cell.data,
            volume.data,
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
