import tempfile, os
import numpy as np
from scipy.io.netcdf import netcdf_file
from .dummy_controller import DummyController


class netCDF:
    def __init__(self, storage, settings, simulator):
        self.storage = storage
        self.settings = settings
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
        ncdf.createDimension("T", len(self.settings.steps))
        ncdf.createDimension("X", self.settings.grid[0])
        ncdf.createDimension("Z", self.settings.grid[1])
        ncdf.createDimension("Volume", len(self.settings.v_bins) - 1)

    def _create_variables(self, ncdf):
        self.vars["T"] = ncdf.createVariable("T", "f", ["T"])
        self.vars["T"].units = "seconds"

        self.vars["X"] = ncdf.createVariable("X", "f", ["X"])
        self.vars["X"][:] = (self.settings.size[0] / self.settings.grid[0]) * (1 / 2 + np.arange(self.settings.grid[0]))
        self.vars["X"].units = "metres"

        self.vars["Z"] = ncdf.createVariable("Z", "f", ["Z"])
        self.vars["Z"][:] = (self.settings.size[1] / self.settings.grid[1]) * (1 / 2 + np.arange(self.settings.grid[1]))
        self.vars["Z"].units = "metres"

        for var in self.simulator.products.keys():
            # TODO: write unit, description
            dimensions = ("T", "X", "Z", "Volume") if len(self.simulator.products[var].shape) == 3 else ("T", "X", "Z")
            self.vars[var] = ncdf.createVariable(var, "f", dimensions)

    def _write_variables(self, i):
        self.vars["T"][i] = self.settings.steps[i] * self.settings.dt
        for var in self.simulator.products.keys():
            if len(self.simulator.products[var].shape) == 3:
                self.vars[var][i, :, :, :] = self.storage.load(self.settings.steps[i], var)
            else:
                self.vars[var][i, :, :] = self.storage.load(self.settings.steps[i], var)

    def run(self, controller=None):
        if controller is None:
            controller = DummyController()
        with controller:
            controller.set_percent(0)
            with netcdf_file(self.tempfile_fd, mode='w') as ncdf:
                self._create_dimensions(ncdf)
                self._create_variables(ncdf)
                for i in range(len(self.settings.steps)):
                    if controller.panic:
                        break
                    self._write_variables(i)
                    controller.set_percent((i+1) / len(self.settings.steps))
