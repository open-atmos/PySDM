""" common code for products computing statistical moments (e.g., effective radius, acidity) """

from abc import ABC

import numpy as np

from PySDM.products.impl.product import Product


class MomentProduct(Product, ABC):
    def __init__(self, name, unit):
        super().__init__(name=name, unit=unit)
        self.moment_0 = None
        self.moments = None

    def register(self, builder):
        super().register(builder)
        self.moment_0 = self.particulator.Storage.empty(
            self.particulator.mesh.n_cell, dtype=float
        )
        self.moments = self.particulator.Storage.empty(
            (1, self.particulator.mesh.n_cell), dtype=float
        )

    def _download_moment_to_buffer(
        self,
        *,
        attr,
        rank,
        filter_attr="water mass",
        filter_range=(-np.inf, np.inf),
        weighting_attribute="water mass",
        weighting_rank=0,
        skip_division_by_m0=False,
    ):
        self.particulator.moments(
            moment_0=self.moment_0,
            moments=self.moments,
            specs={attr: (rank,)},
            attr_name=filter_attr,
            attr_range=filter_range,
            weighting_attribute=weighting_attribute,
            weighting_rank=weighting_rank,
            skip_division_by_m0=skip_division_by_m0,
        )
        if rank == 0:  # TODO #217
            self._download_to_buffer(self.moment_0)
        else:
            self._download_to_buffer(self.moments[0, :])
