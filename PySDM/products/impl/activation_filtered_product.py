"""
common base class for products filtering droplets based on their activation state
"""

import numpy as np


class _ActivationFilteredProduct:
    def __init__(
        self,
        *,
        count_unactivated: bool,
        count_activated: bool,
    ):
        self.__filter_attr = "wet to critical volume ratio"
        self.__filter_range = [0, np.inf]
        if not count_activated:
            self.__filter_range[1] = 1
        if not count_unactivated:
            self.__filter_range[0] = 1

    def impl(self, *, attr, rank):
        getattr(self, "_download_moment_to_buffer")(
            attr=attr,
            rank=rank,
            filter_attr=self.__filter_attr,
            filter_range=self.__filter_range,
        )

    def register(self, builder):
        builder.request_attribute(self.__filter_attr)
