import numpy as np
from ..environments._moist import _Moist


class Product:

    def __init__(self, name, unit=None, description=None):
        self.name = name
        self.unit = unit
        self.description = description
        self.shape = None
        self.buffer = None
        self.particulator = None

    def register(self, builder):
        self.particulator = builder.particulator
        self.formulae = self.particulator.formulae
        self.shape = self.particulator.mesh.grid
        self.buffer = np.empty(self.particulator.mesh.grid)

    def download_to_buffer(self, storage):
        storage.download(self.buffer.ravel())


class MomentProduct(Product):

    def __init__(self, name, unit, description):
        super().__init__(name, unit, description)
        self.moment_0 = None
        self.moments = None

    def register(self, builder):
        super().register(builder)
        self.moment_0 = self.particulator.Storage.empty(self.particulator.mesh.n_cell, dtype=float)
        self.moments = self.particulator.Storage.empty((1, self.particulator.mesh.n_cell), dtype=float)

    def download_moment_to_buffer(self, attr, rank, filter_attr='volume', filter_range=(-np.inf, np.inf),
                                  weighting_attribute='volume', weighting_rank=0):
        self.particulator.attributes.moments(self.moment_0, self.moments, {attr: (rank,)},
                                             attr_name=filter_attr, attr_range=filter_range,
                                             weighting_attribute=weighting_attribute, weighting_rank=weighting_rank)
        if rank == 0:  # TODO #217
            self.download_to_buffer(self.moment_0)
        else:
            self.download_to_buffer(self.moments[0, :])


class SpectrumMomentProduct(Product):
    def __init__(self, name, unit, description):
        super().__init__(name, unit, description)
        self.attr_bins_edges = None
        self.moment_0 = None
        self.moments = None

    def register(self, builder):
        super().register(builder)
        self.moment_0 = self.particulator.Storage.empty((len(self.attr_bins_edges) - 1, self.particulator.mesh.n_cell), dtype=float)
        self.moments = self.particulator.Storage.empty((len(self.attr_bins_edges) - 1, self.particulator.mesh.n_cell), dtype=float)

    def recalculate_spectrum_moment(self, attr, rank, filter_attr='volume',
                                            weighting_attribute='volume', weighting_rank=0):
        self.particulator.attributes.spectrum_moments(self.moment_0, self.moments, attr, rank, self.attr_bins_edges,
                                                      attr_name=filter_attr,
                                                      weighting_attribute=weighting_attribute, weighting_rank=weighting_rank)

    def download_spectrum_moment_to_buffer(self, rank, bin_number):
        if rank == 0:  # TODO #217
            self.download_to_buffer(self.moment_0[bin_number, :])
        else:
            self.download_to_buffer(self.moments[bin_number, :])


class MoistEnvironmentProduct(Product):
    def __init__(self, **args):
        super().__init__(**args)
        self._name = self.name
        self.name = self._name + '_env'
        self.environment = None
        self.source = None

    def register(self, builder):
        assert isinstance(builder.particulator.env, _Moist)
        super().register(builder)
        self.particulator.observers.append(self)
        self.environment = builder.particulator.env
        self.source = self.environment[self._name]

    def notify(self):
        self.source = self.environment.get_predicted(self._name)

    def get(self):
        self.download_to_buffer(self.source)
        return self.buffer

