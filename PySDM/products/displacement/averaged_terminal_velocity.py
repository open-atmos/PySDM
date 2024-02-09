"""
average terminal with weighting either by number or volume
"""

import numpy as np

from PySDM.products.impl.moment_product import MomentProduct


class AveragedTerminalVelocity(MomentProduct):
    def __init__(
        self, *, radius_range=(0, np.inf), weighting="volume", unit="m/s", name=None
    ):
        self.attr = "terminal velocity"

        if weighting == "number":
            self.weighting_rank = 0
        elif weighting == "volume":
            self.weighting_rank = 1
        else:
            raise NotImplementedError()

        self.radius_range = radius_range
        super().__init__(name=name, unit=unit)

    def register(self, builder):
        builder.request_attribute(self.attr)
        super().register(builder)

    def _impl(self, **kwargs):
        self._download_moment_to_buffer(
            attr=self.attr,
            rank=1,
            filter_range=(
                self.formulae.trivia.volume(self.radius_range[0]),
                self.formulae.trivia.volume(self.radius_range[1]),
            ),
            weighting_attribute="volume",
            weighting_rank=self.weighting_rank,
        )

        return self.buffer
