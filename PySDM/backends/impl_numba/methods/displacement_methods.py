import numba
from PySDM.backends.impl_numba import conf
from ...impl_common.backend_methods import BackendMethods


@numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
# pylint: disable=too-many-arguments
def calculate_displacement_body_common(
        dim, droplet, scheme, _l, _r, displacement, courant, position_in_cell
):
    omega = position_in_cell[dim, droplet]
    displacement[dim, droplet] = scheme(omega, courant[_l], courant[_r])


class DisplacementMethods(BackendMethods):
    @staticmethod
    @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False, 'cache': False}})
    # pylint: disable=too-many-arguments
    def calculate_displacement_body_1d(dim, scheme, displacement, courant,
                                       cell_origin, position_in_cell):
        length = displacement.shape[1]
        for droplet in numba.prange(length):  # pylint: disable=not-an-iterable
            # Arakawa-C grid
            _l = cell_origin[0, droplet]
            _r = cell_origin[0, droplet] + 1
            calculate_displacement_body_common(dim, droplet, scheme, _l, _r,
                                               displacement, courant, position_in_cell)

    @staticmethod
    @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False, 'cache': False}})
    # pylint: disable=too-many-arguments
    def calculate_displacement_body_2d(dim, scheme, displacement, courant,
                                       cell_origin, position_in_cell):
        length = displacement.shape[1]
        for droplet in numba.prange(length):  # pylint: disable=not-an-iterable
            # Arakawa-C grid
            _l = (cell_origin[0, droplet],
                  cell_origin[1, droplet])
            _r = (cell_origin[0, droplet] + 1 * (dim == 0),
                  cell_origin[1, droplet] + 1 * (dim == 1))
            calculate_displacement_body_common(dim, droplet, scheme, _l, _r,
                                               displacement, courant, position_in_cell)

    # pylint: disable=too-many-arguments
    def calculate_displacement(self, dim, displacement, courant, cell_origin, position_in_cell):
        n_dims = len(courant.shape)
        scheme = self.formulae.particle_advection.displacement
        if n_dims == 1:
            DisplacementMethods.calculate_displacement_body_1d(
                dim, scheme, displacement.data, courant.data,
                cell_origin.data, position_in_cell.data
            )
        elif n_dims == 2:
            DisplacementMethods.calculate_displacement_body_2d(
                dim, scheme, displacement.data, courant.data,
                cell_origin.data, position_in_cell.data
            )
        else:
            raise NotImplementedError()

    @staticmethod
    @numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
    # pylint: disable=too-many-arguments
    def flag_precipitated_body(
        cell_origin, position_in_cell, volume, multiplicity, idx, length, healthy
    ):
        rainfall = 0.
        flag = len(idx)
        for i in range(length):
            if cell_origin[-1, idx[i]] + position_in_cell[-1, idx[i]] < 0:
                rainfall += volume[idx[i]] * multiplicity[idx[i]]
                idx[i] = flag
                healthy[0] = 0
        return rainfall

    @staticmethod
    # pylint: disable=too-many-arguments
    def flag_precipitated(
        cell_origin, position_in_cell, volume, multiplicity, idx, length, healthy
    ) -> float:
        return DisplacementMethods.flag_precipitated_body(
            cell_origin.data, position_in_cell.data, volume.data,
            multiplicity.data, idx.data, length, healthy.data)
