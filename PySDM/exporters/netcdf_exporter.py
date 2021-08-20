import numpy as np
from scipy.io.netcdf import netcdf_file


class NetCDFExporter:
    def __init__(self, storage, settings, simulator, filename):
        self.storage = storage
        self.settings = settings
        self.simulator = simulator
        self.vars = None
        self.filename = filename
        self.XZ = ('X', 'Z')

    def _write_settings(self, ncdf):
        for setting in dir(self.settings):
            setattr(ncdf, setting, getattr(self.settings, setting))

    def _create_dimensions(self, ncdf):
        ncdf.createDimension("T", len(self.settings.output_steps))
        for index, label in enumerate(self.XZ):
            ncdf.createDimension(label, self.settings.grid[index])
        ncdf.createDimension("ParticleVolume",
                             len(self.settings.formulae.trivia.volume(self.settings.r_bins_edges)) - 1)

    def _create_variables(self, ncdf):
        self.vars = {}
        self.vars["T"] = ncdf.createVariable("T", "f", ["T"])
        self.vars["T"].units = "seconds"

        for index, label in enumerate(self.XZ):
            self.vars[label] = ncdf.createVariable(label, "f", (label,))
            self.vars[label][:] = ((self.settings.size[index] / self.settings.grid[index])
                                   * (1 / 2 + np.arange(self.settings.grid[index])))
            self.vars[label].units = "metres"

        # TODO #340 ParticleVolume var

        for name, instance in self.simulator.products.items():
            assert name not in self.vars

            n_dimensions = len(instance.shape)
            if n_dimensions == 3:
                dimensions = ("T", "X", "Z", "ParticleVolume")
            elif n_dimensions == 2:
                dimensions = ("T", "X", "Z")
            elif n_dimensions == 0:
                dimensions = ("T",)
            else:
                raise NotImplementedError()
            self.vars[name] = ncdf.createVariable(name, "f", dimensions)
            self.vars[name].units = instance.unit
            self.vars[name].long_name = instance.description

    def _write_variables(self, i):
        self.vars["T"][i] = self.settings.output_steps[i] * self.settings.dt
        for var in self.simulator.products.keys():
            n_dimensions = len(self.simulator.products[var].shape)
            if n_dimensions == 3:
                self.vars[var][i, :, :, :] = self.storage.load(var, self.settings.output_steps[i])
            elif n_dimensions == 2:
                self.vars[var][i, :, :] = self.storage.load(var, self.settings.output_steps[i])
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
            with netcdf_file(self.filename, mode='w') as ncdf:
                self._write_settings(ncdf)
                self._create_dimensions(ncdf)
                self._create_variables(ncdf)
                for i in range(len(self.settings.output_steps)):
                    if controller.panic:
                        break
                    self._write_variables(i)
                    controller.set_percent((i+1) / len(self.settings.output_steps))
