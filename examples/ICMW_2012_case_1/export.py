import netCDF4, tempfile, os
import numpy as np
from scipy.io.netcdf import netcdf_file

class netCDF:
    def __init__(self, storage, setup):
        self.storage = storage
        self.setup = setup
        self.vars = {}

        self.tempfile_fd, self.tempfile_path = tempfile.mkstemp(
            dir=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'output'),
            suffix='.nc'
        )

    @property
    def filename(self):
        return str(os.path.join('output', os.path.basename(self.tempfile_path)))

    def _createDimensions(self, ncdf):
        ncdf.createDimension("T", len(self.setup.steps))
        ncdf.createDimension("X", self.setup.grid[0])
        ncdf.createDimension("Z", self.setup.grid[1])

    def _createVariables(self, ncdf):
        self.vars["T"] = ncdf.createVariable("T", "f", ["T"])
        self.vars["T"].units = "seconds"

        self.vars["X"] = ncdf.createVariable("X", "f", ["X"])
        self.vars["X"][:] = self.setup.dx * (1/2 + np.arange(self.setup.grid[0]))
        self.vars["X"].units = "metres"

        self.vars["Z"] = ncdf.createVariable("Z", "f", ["Z"])
        self.vars["Z"][:] = self.setup.dz * (1/2 + np.arange(self.setup.grid[1]))
        self.vars["Z"].units = "metres"

        for var in self.setup.output_vars:
            self.vars[var] = ncdf.createVariable(var, "f", ["T", "X", "Z"])

    def _writeVariables(self, ncdf, i):
        self.vars["T"][i] = self.setup.steps[i] * self.setup.dt
        for var in self.setup.output_vars:
            self.vars[var][i, :, :] = self.storage.load(self.setup.steps[i], var)

    def run(self, controller):
        with controller:
            with netcdf_file(self.tempfile_fd, mode='w') as ncdf:
                self._createDimensions(ncdf)
                self._createVariables(ncdf)
                for i in range(len(self.setup.steps)):
                    if controller.panic:
                        break
                    self._writeVariables(ncdf, i)
                    controller.set_percent(i / len(self.setup.steps))
