"""
factory for arbitrary-moment product classes
"""
from PySDM.products.impl.moment_product import MomentProduct


def make_arbitrary_moment_product(**kwargs):
    for arg in kwargs:
        assert arg in ("rank", "attr", "attr_unit")

    class ArbitraryMoment(MomentProduct):
        def __init__(
            self, name=None, unit=f"({kwargs['attr_unit']})**{kwargs['rank']}"
        ):
            super().__init__(name=name, unit=unit)
            self.attr = kwargs["attr"]
            self.rank = kwargs["rank"]

        def _impl(self, **kwargs):
            self._download_moment_to_buffer(
                attr=self.attr, rank=self.rank, skip_division_by_m0=True
            )
            return self.buffer

    return ArbitraryMoment


VolumeFirstMoment = make_arbitrary_moment_product(
    rank=1, attr="volume", attr_unit="m^3"
)

RadiusSixthMoment = make_arbitrary_moment_product(rank=6, attr="radius", attr_unit="m")

RadiusFirstMoment = make_arbitrary_moment_product(rank=1, attr="radius", attr_unit="m")
