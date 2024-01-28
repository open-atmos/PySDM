"""
VTK exporter implemented using [pyevtk](https://pypi.org/project/pyevtk/)
"""

import numbers
import os
import sys

import numpy as np
from pyevtk.hl import gridToVTK, pointsToVTK
from pyevtk.vtk import VtkGroup


class VTKExporter:
    """
    Example of use:

    exporter = VTKExporter()

    for step in range(settings.n_steps):
        simulation.particulator.run(1)

        exporter.export_attributes(simulation.particulator)
        exporter.export_products(simulation.particulator)

    """

    def __init__(
        self,
        *,
        path=".",
        attributes_filename="sd_attributes",
        products_filename="sd_products",
        file_num_len=10,
        verbose=False,
    ):
        self.path = os.path.join(path, "output")

        if not os.path.isdir(self.path):
            os.mkdir(self.path)

        self.attributes_file_path = os.path.join(self.path, attributes_filename)
        self.products_file_path = os.path.join(self.path, products_filename)
        self.num_len = file_num_len
        self.exported_times = {}
        self.exported_times["attributes"] = {}
        self.exported_times["products"] = {}
        self.verbose = verbose

    def write_pvd(self):
        pvd_attributes = VtkGroup(self.attributes_file_path)
        for k, v in self.exported_times["attributes"].items():
            pvd_attributes.addFile(k + ".vtu", sim_time=v)
        pvd_attributes.save()

        pvd_products = VtkGroup(self.products_file_path)
        for k, v in self.exported_times["products"].items():
            pvd_products.addFile(k + ".vts", sim_time=v)
        pvd_products.save()

    def export_attributes(self, particulator):
        path = (
            self.attributes_file_path
            + "_num"
            + self.add_leading_zeros(particulator.n_steps)
        )
        self.exported_times["attributes"][path] = particulator.n_steps * particulator.dt
        if self.verbose:
            print("Exporting Attributes to vtk, path: " + path)
        payload = {}

        for k in particulator.attributes.keys():
            if len(particulator.attributes[k].shape) != 1:
                tmp = particulator.attributes[k].to_ndarray(raw=True)
                tmp_dict = {
                    k + "[" + str(i) + "]": tmp[i]
                    for i in range(len(particulator.attributes[k].shape))
                }

                payload.update(tmp_dict)
            else:
                payload[k] = particulator.attributes[k].to_ndarray(raw=True)

        payload.update(
            {
                k: np.array(v)
                for k, v in payload.items()
                if not (v.flags["C_CONTIGUOUS"] or v.flags["F_CONTIGUOUS"])
            }
        )

        if particulator.mesh.dimension == 2:
            y = (
                particulator.mesh.size[0]
                / particulator.mesh.grid[0]
                * (payload["cell origin[0]"] + payload["position in cell[0]"])
            )
            x = (
                particulator.mesh.size[1]
                / particulator.mesh.grid[1]
                * (payload["cell origin[1]"] + payload["position in cell[1]"])
            )
            z = np.full_like(x, 0)
        else:
            raise NotImplementedError(
                "Only 2 dimensions array is supported at the moment."
            )

        pointsToVTK(path, x, y, z, data=payload)

    def export_products(self, particulator):
        if len(particulator.products) != 0:
            path = (
                self.products_file_path
                + "_num"
                + self.add_leading_zeros(particulator.n_steps)
            )
            self.exported_times["products"][path] = (
                particulator.n_steps * particulator.dt
            )
            if self.verbose:
                print("Exporting Products to vtk, path: " + path)
            payload = {}

            if particulator.mesh.dimension != 2:
                raise NotImplementedError(
                    "Only 2 dimensions data is supported at the moment."
                )

            data_shape = (particulator.mesh.grid[1], particulator.mesh.grid[0], 1)

            for k in particulator.products.keys():
                v = particulator.products[k].get()

                if isinstance(v, np.ndarray):
                    if v.shape == particulator.mesh.grid:
                        payload[k] = v[:, :, np.newaxis]
                    else:
                        if self.verbose:
                            print(
                                f"{k} shape {v.shape} not equals data shape {data_shape}"
                                f" and will not be exported",
                                file=sys.stderr,
                            )
                elif isinstance(v, numbers.Number):
                    if self.verbose:
                        print(
                            f"{k} is a Number and will not be exported", file=sys.stderr
                        )
                else:
                    if self.verbose:
                        print(f"{k} export is not possible", file=sys.stderr)

            y, x, z = np.mgrid[
                : particulator.mesh.grid[0] + 1, : particulator.mesh.grid[1] + 1, :1
            ]
            y = y * particulator.mesh.size[0] / particulator.mesh.grid[0]
            x = x * particulator.mesh.size[1] / particulator.mesh.grid[1]
            z = z * 1.0

            gridToVTK(path, x, y, z, cellData=payload)
        else:
            if self.verbose:
                print("No products to export")

    def add_leading_zeros(self, a):
        return "".join(["0" for i in range(self.num_len - len(str(a)))]) + str(a)
