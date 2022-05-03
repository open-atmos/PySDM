"""
CPU implementation of backend methods for particle displacement (advection and sedimentation)
"""
import numba

from PySDM.backends.impl_numba import conf

from ...impl_common.backend_methods import BackendMethods


@numba.njit(**{**conf.JIT_FLAGS, **{"parallel": False}})
# pylint: disable=too-many-arguments
def calculate_displacement_body_common(
    dim, droplet, scheme, _l, _r, displacement, courant, position_in_cell, n_substeps
):
    omega = position_in_cell[dim, droplet]
    displacement[dim, droplet] = scheme(
        omega, courant[_l] / n_substeps, courant[_r] / n_substeps
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
            _l = (cell_origin[0, droplet], cell_origin[1, droplet])
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
        else:
            raise NotImplementedError()

    @staticmethod
    @numba.njit(**{**conf.JIT_FLAGS, **{"parallel": False}})
    # pylint: disable=too-many-arguments
    def flag_precipitated_body(
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

    @staticmethod
    @numba.njit(**{**conf.JIT_FLAGS, **{"parallel": False}})
    # pylint: disable=too-many-arguments
    def flag_out_of_column_body(
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

    @staticmethod
    # pylint: disable=too-many-arguments
    def flag_precipitated(
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
        return DisplacementMethods.flag_precipitated_body(
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

    @staticmethod
    # pylint: disable=too-many-arguments
    def flag_out_of_column(
        cell_origin, position_in_cell, idx, length, healthy, domain_top_level_index
    ) -> float:
        return DisplacementMethods.flag_out_of_column_body(
            cell_origin.data,
            position_in_cell.data,
            idx.data,
            length,
            healthy.data,
            domain_top_level_index,
        )
