"""
average pH (averaging after or before taking the logarithm in pH definition)
with weighting either by number or volume
"""

import numpy as np

from PySDM.products.impl.moment_product import MomentProduct


class Acidity(MomentProduct):
    def __init__(
        self,
        *,
        radius_range=(0, np.inf),
        weighting="volume",
        attr="conc_H",
        unit="dimensionless",
        name=None
    ):
        assert attr in ("pH", "moles_H", "conc_H")
        self.attr = attr

        if weighting == "number":
            self.weighting_rank = 0
        elif weighting == "volume":
            self.weighting_rank = 1
        else:
            raise NotImplementedError()

        self.radius_range = radius_range
        super().__init__(name=name, unit=unit)

    def register(self, builder):
        builder.request_attribute("conc_H")
        super().register(builder)

    def _impl(self, **kwargs):
        self._download_moment_to_buffer(
            attr=self.attr,
            rank=1,
            filter_range=(
                self.formulae.trivia.volume(self.radius_range[0]),
                self.formulae.trivia.volume(self.radius_range[1]),
            ),
            filter_attr="volume",
            weighting_attribute="volume",
            weighting_rank=self.weighting_rank,
        )
        if self.attr == "conc_H":
            self.buffer[:] = self.formulae.trivia.H2pH(self.buffer[:])
        elif self.attr == "pH":
            pass
        else:
            raise NotImplementedError()

        return self.buffer
