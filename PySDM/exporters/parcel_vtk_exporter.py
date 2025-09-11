"""
VTK Exporter for Pyrcel PySDM simulations.

This module defines `VTKExporterPyrcel`, a subclass of `PySDM.exporters.VTKExporter`,
that writes simulation outputs to VTK format using `pyevtk`. It exports product
profiles (e.g., relative humidity) as unstructured grids and particle attributes
as point clouds, along with `.pvd` collection files for time-series visualization
in ParaView.
"""

from pyevtk.hl import unstructuredGridToVTK, pointsToVTK
from pyevtk.vtk import VtkHexahedron, VtkGroup
import numpy as np

from PySDM.exporters import VTKExporter


class VTKExporterPyrcel(VTKExporter):
    """
    Custom VTK exporter for Pyrcel PySDM, exporting products as grids
    and attributes as point clouds for ParaView visualization.
    """

    def __init__(self, n_sd):
        super().__init__()
        self.x = np.random.random(n_sd)  # pylint: disable=invalid-name
        self.y = np.random.random(n_sd)  # pylint: disable=invalid-name
        self.z = np.random.random(n_sd)  # pylint: disable=invalid-name

    def write_pvd(self):
        pvd_attributes = VtkGroup(self.attributes_file_path)
        for key, value in self.exported_times["attributes"].items():
            pvd_attributes.addFile(key + ".vtu", sim_time=value)
        pvd_attributes.save()

        pvd_products = VtkGroup(self.products_file_path)
        for key, value in self.exported_times["products"].items():
            pvd_products.addFile(key + ".vtu", sim_time=value)
        pvd_products.save()

    def export_products(
        self, output, step, simulation
    ):  # pylint: disable=arguments-differ, too-many-locals
        path = self.products_file_path + "_num" + self.add_leading_zeros(step)

        self.exported_times["products"][path] = step * simulation.particulator.dt

        n_levels = len(output["products"]["z"])
        n_vertices = 4 * n_levels
        x = np.zeros(n_vertices)  # pylint: disable=invalid-name
        y = np.zeros(n_vertices)  # pylint: disable=invalid-name
        z = np.zeros(n_vertices)  # pylint: disable=invalid-name
        conn = [0, 1, 2, 3]
        ctype = np.zeros(n_levels - 1)
        ctype[:] = VtkHexahedron.tid

        _volume = (
            simulation.particulator.environment.mass_of_dry_air
            / output["products"]["rhod"][0]
        )
        _dz = output["products"]["z"][1] - output["products"]["z"][0]
        for level in range(n_levels):
            _z = output["products"]["z"][level]
            if level > 0:
                prev_to_curr_density_ratio = (
                    output["products"]["rhod"][level - 1]
                    / output["products"]["rhod"][level]
                )
                _volume *= prev_to_curr_density_ratio
                _dz = (
                    output["products"]["z"][level] - output["products"]["z"][level - 1]
                )
            _area = _volume / _dz
            _half_diagonal = (2 * _area) ** 0.5
            x[i := level * 4], y[i], z[i] = -_half_diagonal, -_half_diagonal, _z
            x[i := i + 1], y[i], z[i] = -_half_diagonal, _half_diagonal, _z
            x[i := i + 1], y[i], z[i] = _half_diagonal, _half_diagonal, _z
            x[i := i + 1], y[i], z[i] = _half_diagonal, -_half_diagonal, _z
            conn += [*range(4 * (level + 1), 4 * (level + 2))] * 2
        conn = np.asarray(conn[:-4])
        offset = np.asarray(range(8, 8 * n_levels, 8))

        _ = {"test_pd": np.array([44] * n_vertices)}  # pointData

        _RH = output["products"]["S_max_percent"]  # pylint: disable=invalid-name
        cell_data = {"RH": np.full(shape=(len(_RH) - 1,), fill_value=np.nan)}
        cell_data["RH"][:step] = (np.array(_RH[:-1] + np.diff(_RH) / 2))[:step]
        unstructuredGridToVTK(
            path,
            x,
            y,
            z,
            connectivity=conn,
            offsets=offset,
            cell_types=ctype,
            cellData=cell_data,
            # pointData=point_data,
            # fieldData=field_data,
        )

    def export_attributes(
        self, output, step, simulation
    ):  # pylint: disable=arguments-differ
        path = self.attributes_file_path + "_num" + self.add_leading_zeros(step)
        self.exported_times["attributes"][path] = step * simulation.particulator.dt

        payload = {}

        for k in output["attributes"].keys():
            payload[k] = np.asarray(output["attributes"][k])[:, step].copy()

        if step != 0:
            dz = (
                output["products"]["z"][step] - output["products"]["z"][step - 1]
            )  # pylint: disable=invalid-name
            if step == 1:
                self.z *= dz
            else:
                self.z += dz

        print(pointsToVTK(path, self.x, self.y, self.z, data=payload))
