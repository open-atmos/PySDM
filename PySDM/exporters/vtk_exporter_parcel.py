"""
VTK Exporter for parcel PySDM simulations.

This module defines `VTKExporterParcel`, a subclass of `PySDM.exporters.VTKExporter`,
that writes simulation outputs to VTK format using `pyevtk`. It exports product
profiles (e.g., relative humidity) as unstructured grids and particle attributes
as point clouds, along with `.pvd` collection files for time-series visualization
in ParaView.
"""

from pyevtk.hl import unstructuredGridToVTK, pointsToVTK
from pyevtk.vtk import VtkHexahedron, VtkGroup
import numpy as np

from PySDM.exporters.vtk_exporter import VTKExporter


# pylint: disable=too-many-instance-attributes
class VTKExporterParcel(VTKExporter):
    """
    Custom VTK exporter for parcel PySDM, exporting products as grids
    and attributes as point clouds for ParaView visualization.
    """

    def __init__(self, n_sd, output, mass_of_dry_air):
        super().__init__()
        self.output = output
        self.coords = {
            "x": np.random.random(n_sd),
            "y": np.random.random(n_sd),
            "z": np.random.random(n_sd),
        }
        self.half_diagonal = []
        self.n_levels = len(self.output["products"]["z"])

        volume = mass_of_dry_air / output["products"]["rhod"][0]
        delta_z = output["products"]["z"][1] - output["products"]["z"][0]
        for level in range(self.n_levels):
            if level > 0:
                prev_to_curr_density_ratio = (
                    output["products"]["rhod"][level - 1]
                    / output["products"]["rhod"][level]
                )
                volume *= prev_to_curr_density_ratio
                delta_z = (
                    output["products"]["z"][level] - output["products"]["z"][level - 1]
                )
            area = volume / delta_z
            self.half_diagonal.append((2 * area) ** 0.5)

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
        self, step, simulation
    ):  # pylint: disable=arguments-differ, too-many-locals
        path = self.products_file_path + "_num" + self.add_leading_zeros(step)
        self.exported_times["products"][path] = step * simulation.particulator.dt

        n_vertices = 4 * self.n_levels
        x_vertices = np.zeros(n_vertices)
        y_vertices = np.zeros(n_vertices)
        z_vertices = np.zeros(n_vertices)
        connectivity = [0, 1, 2, 3]
        cell_type = np.zeros(self.n_levels - 1)
        cell_type[:] = VtkHexahedron.tid

        for level in range(self.n_levels):
            half_diag = self.half_diagonal[level]
            current_z = self.output["products"]["z"][level]
            idx = level * 4
            x_vertices[idx], y_vertices[idx], z_vertices[idx] = (
                -half_diag,
                -half_diag,
                current_z,
            )
            idx += 1
            x_vertices[idx], y_vertices[idx], z_vertices[idx] = (
                -half_diag,
                half_diag,
                current_z,
            )
            idx += 1
            x_vertices[idx], y_vertices[idx], z_vertices[idx] = (
                half_diag,
                half_diag,
                current_z,
            )
            idx += 1
            x_vertices[idx], y_vertices[idx], z_vertices[idx] = (
                half_diag,
                -half_diag,
                current_z,
            )
            connectivity += [*range(4 * (level + 1), 4 * (level + 2))] * 2
        connectivity = np.asarray(connectivity[:-4])
        offset = np.asarray(range(8, 8 * self.n_levels, 8))

        _ = {"test_pd": np.array([44] * n_vertices)}

        relative_humidity = self.output["products"]["S_max_percent"]
        cell_data = {
            "RH": np.full(shape=(len(relative_humidity) - 1,), fill_value=np.nan)
        }
        cell_data["RH"][:step] = (
            np.array(relative_humidity[:-1] + np.diff(relative_humidity) / 2)
        )[:step]
        unstructuredGridToVTK(
            path,
            x_vertices,
            y_vertices,
            z_vertices,
            connectivity=connectivity,
            offsets=offset,
            cell_types=cell_type,
            cellData=cell_data,
        )

    def export_attributes(self, step, simulation):  # pylint: disable=arguments-differ
        path = self.attributes_file_path + "_num" + self.add_leading_zeros(step)
        self.exported_times["attributes"][path] = step * simulation.particulator.dt

        payload = {}

        for key in self.output["attributes"].keys():
            payload[key] = np.asarray(self.output["attributes"][key])[:, step].copy()
        if step != 0:
            delta_z = (
                self.output["products"]["z"][step]
                - self.output["products"]["z"][step - 1]
            )
            if step == 1:
                self.coords["z"] *= delta_z
            else:
                self.coords["z"] += delta_z

        pointsToVTK(
            path,
            2 * (self.coords["x"] - 0.5) * self.half_diagonal[step],
            2 * (self.coords["y"] - 0.5) * self.half_diagonal[step],
            self.coords["z"],
            data=payload,
        )
