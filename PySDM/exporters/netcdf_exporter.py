""" netCDF exporter implemented using
 [SciPy.io.netcdf_file](https://docs.scipy.org/doc/scipy/reference/tutorial/io.html#netcdf)
"""

import numpy as np
from scipy.io import netcdf_file

from PySDM.products.impl.spectrum_moment_product import SpectrumMomentProduct

DIM_SUFFIX = "_bin_left_edges"


class NetCDFExporter:  # pylint: disable=too-few-public-methods
    def __init__(self, storage, settings, simulator, filename):
        self.storage = storage
        self.settings = settings
        self.simulator = simulator
        self.vars = None
        self.filename = filename
        self.XZ = ("X", "Z")

    def _write_settings(self, ncdf):
        for setting in dir(self.settings):
            setattr(ncdf, setting, getattr(self.settings, setting))

    def _create_dimensions(self, ncdf):
        ncdf.createDimension("T", len(self.settings.output_steps))

        for index, label in enumerate(self.XZ):
            ncdf.createDimension(label, self.settings.grid[index])

        for name, instance in self.simulator.products.items():
            if isinstance(instance, SpectrumMomentProduct):
                ncdf.createDimension(
                    f"{name}{DIM_SUFFIX}", len(instance.attr_bins_edges) - 1
                )

    def _create_variables(self, ncdf):
        self.vars = {}
        self.vars["T"] = ncdf.createVariable("T", "f", ["T"])
        self.vars["T"].units = "seconds"

        for index, label in enumerate(self.XZ):
            self.vars[label] = ncdf.createVariable(label, "f", (label,))
            self.vars[label][:] = (
                self.settings.size[index] / self.settings.grid[index]
            ) * (1 / 2 + np.arange(self.settings.grid[index]))
            self.vars[label].units = "metres"

        for name, instance in self.simulator.products.items():
            if isinstance(instance, SpectrumMomentProduct):
                label = f"{name}{DIM_SUFFIX}"
                self.vars[label] = ncdf.createVariable(label, "f", (label,))
                self.vars[label][:] = instance.attr_bins_edges.to_ndarray()[:-1]
                self.vars[label].units = instance.attr_unit

        for name, instance in self.simulator.products.items():
            if name in self.vars:
                raise AssertionError(
                    f"product ({name}) has same name as one of netCDF dimensions"
                )

            n_dimensions = len(instance.shape)
            if n_dimensions == 3:
                dimensions = ("T", "X", "Z", f"{name}{DIM_SUFFIX}")
            elif n_dimensions == 2:
                dimensions = ("T", "X", "Z")
            elif n_dimensions == 0:
                dimensions = ("T",)
            else:
                raise NotImplementedError()
            self.vars[name] = ncdf.createVariable(name, "f", dimensions)
            self.vars[name].units = instance.unit

    def _write_variables(self, i):
        self.vars["T"][i] = self.settings.output_steps[i] * self.settings.dt
        for var in self.simulator.products.keys():
            n_dimensions = len(self.simulator.products[var].shape)
            if n_dimensions == 3:
                self.vars[var][i, :, :, :] = self.storage.load(
                    var, self.settings.output_steps[i]
                )
            elif n_dimensions == 2:
                self.vars[var][i, :, :] = self.storage.load(
                    var, self.settings.output_steps[i]
                )
            elif n_dimensions == 0:
                if i == 0:
                    self.vars[var][:] = self.storage.load(var)
                else:
                    pass
            else:
                raise NotImplementedError()

    def run(self, controller):
        with controller:
            controller.set_percent(0)
            with netcdf_file(self.filename, mode="w") as ncdf:
                self._write_settings(ncdf)
                self._create_dimensions(ncdf)
                self._create_variables(ncdf)
                for i in range(len(self.settings.output_steps)):
                    if controller.panic:
                        break
                    self._write_variables(i)
                    controller.set_percent((i + 1) / len(self.settings.output_steps))
