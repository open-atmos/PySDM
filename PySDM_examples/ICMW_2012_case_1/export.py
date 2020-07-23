import tempfile, os
import numpy as np
from scipy.io.netcdf import netcdf_file


class netCDF:
    def __init__(self, storage, setup, simulator):
        self.storage = storage
        self.setup = setup
        self.simulator = simulator
        self.vars = {}

        self.tempfile_fd, self.tempfile_path = tempfile.mkstemp(
            dir=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'output'),
            suffix='.nc'
        )

    @property
    def filename(self):
        return str(os.path.join('output', os.path.basename(self.tempfile_path)))

    def _create_dimensions(self, ncdf):
        ncdf.createDimension("T", len(self.setup.steps))
        ncdf.createDimension("X", self.setup.grid[0])
        ncdf.createDimension("Z", self.setup.grid[1])
        ncdf.createDimension("Volume", len(self.setup.v_bins) - 1)

    def _create_variables(self, ncdf):
        self.vars["T"] = ncdf.createVariable("T", "f", ["T"])
        self.vars["T"].units = "seconds"

        self.vars["X"] = ncdf.createVariable("X", "f", ["X"])
        self.vars["X"][:] = (self.setup.size[0] / self.setup.grid[0]) * (1/2 + np.arange(self.setup.grid[0]))
        self.vars["X"].units = "metres"

        self.vars["Z"] = ncdf.createVariable("Z", "f", ["Z"])
        self.vars["Z"][:] = (self.setup.size[1] / self.setup.grid[1]) * (1/2 + np.arange(self.setup.grid[1]))
        self.vars["Z"].units = "metres"

        for var in self.simulator.products.keys():
            # TODO: write unit, description
            dimensions = ("T", "X", "Z", "Volume") if var == "Particles Size Spectrum" else ("T", "X", "Z")
            self.vars[var] = ncdf.createVariable(var, "f", dimensions)

    def _write_variables(self, i):
        self.vars["T"][i] = self.setup.steps[i] * self.setup.dt
        for var in self.simulator.products.keys():
            if var == "Particles Size Spectrum":
                self.vars[var][i, :, :, :] = self.storage.load(self.setup.steps[i], var)
            self.vars[var][i, :, :] = self.storage.load(self.setup.steps[i], var)

    def run(self, controller):
        with controller:
            controller.set_percent(0)
            with netcdf_file(self.tempfile_fd, mode='w') as ncdf:
                self._create_dimensions(ncdf)
                self._create_variables(ncdf)
                for i in range(len(self.setup.steps)):
                    if controller.panic:
                        break
                    self._write_variables(i)
                    controller.set_percent((i+1) / len(self.setup.steps))
