# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest

from PySDM.impl.mesh import Mesh
from PySDM.physics import si


def random_positions(grid: tuple, n_sd: int):
    positions = np.random.random(len(grid) * n_sd).reshape(len(grid), n_sd)
    for dim, span in enumerate(grid):
        positions[dim, :] *= span
    return positions


class TestMesh:
    @staticmethod
    @pytest.mark.parametrize(
        "grid",
        (
            (10,),
            (2, 2),
            (10, 20, 30),
        ),
    )
    def test_strides(grid: tuple):
        # arrange
        n_sd = 666
        positions = random_positions(grid, n_sd)
        size = 44 * np.asarray(grid)
        cell_id_expected, cell_origins, _ = Mesh(grid, size).cellular_attributes(
            positions
        )

        # act
        strides = Mesh._Mesh__strides(grid)

        # assert
        assert len(strides.shape) == 2
        assert strides[-1][-1] == 1
        assert strides.shape == (1, len(grid))

        cell_id_actual = np.array(
            [
                np.dot(strides, cell_origin)  # the recipe for computing cell_id
                for cell_origin in cell_origins.T
            ]
        ).squeeze()
        np.testing.assert_array_equal(cell_id_expected, cell_id_actual)

    @staticmethod
    @pytest.mark.parametrize(
        "mesh",
        (
            Mesh(grid=(10,), size=(2,)),
            Mesh(grid=(2, 2), size=(4, 4)),
            Mesh(grid=(10, 20, 30), size=(5, 5, 5)),
            Mesh(grid=(3, 4, 5, 6), size=(5, 5, 5, 5)),
        ),
    )
    def test_cellular_attributes(mesh):
        # arrange
        n_sd = 666
        positions = random_positions(mesh.grid, n_sd)

        # act
        cell_id, cell_origin, position_in_cell = mesh.cellular_attributes(positions)

        # assert
        assert cell_id.shape == (n_sd,)
        assert cell_origin.shape == (mesh.dimension, n_sd)
        assert position_in_cell.shape == (mesh.dimension, n_sd)

        assert 0 <= min(cell_id) <= max(cell_id) < mesh.n_cell
        assert cell_id.dtype == np.int64

        for dim in range(mesh.dimension):
            assert (
                0
                <= min(cell_origin[dim, :])
                <= max(cell_origin[dim, :])
                < mesh.grid[dim]
            )
        assert cell_origin.dtype == np.int64

        for dim in range(mesh.dimension):
            assert (
                0 <= min(position_in_cell[dim, :]) <= max(position_in_cell[dim, :]) < 1
            )
        assert position_in_cell.dtype == float

    @staticmethod
    @pytest.mark.parametrize(
        "mesh, expected_dv",
        (
            (Mesh.mesh_0d(dv=44 * si.m**3), 44 * si.m**3),
            (Mesh(grid=(100,), size=(1 * si.km,)), 10 * si.m * 1 * si.m * 1 * si.m),
            (
                Mesh(grid=(100, 200), size=(1 * si.km, 2 * si.km)),
                10 * si.m * 10 * si.m * 1 * si.m,
            ),
            (
                Mesh(grid=(10, 20, 30), size=(1 * si.km, 2 * si.km, 3 * si.km)),
                (100 * si.m) ** 3,
            ),
        ),
    )
    def test_dv(mesh, expected_dv):
        assert mesh.dv == expected_dv

    @staticmethod
    @pytest.mark.parametrize(
        "mesh, expected_area",
        (
            (Mesh(grid=(100,), size=(44 * si.km,)), 1 * si.m**2),
            (Mesh(grid=(100, 200), size=(44 * si.km, 666 * si.m)), 44 * si.km * si.m),
        ),
    )
    def test_domain_bottom_surface_area(mesh, expected_area):
        assert mesh.domain_bottom_surface_area == expected_area
