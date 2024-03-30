"""
factory for arbitrary-moment product classes
"""

from PySDM.products.impl.moment_product import MomentProduct


def make_arbitrary_moment_product(**kwargs):
    """returns a product class to be instantiated and passed to a builder"""
    for arg in kwargs:
        assert arg in (
            "rank",
            "attr",
            "attr_unit",
            "skip_division_by_m0",
            "skip_division_by_dv",
        )

    class ArbitraryMoment(MomentProduct):
        def __init__(
            self,
            name=None,
            unit=f"({kwargs['attr_unit']})**{kwargs['rank']}"
            + ("" if kwargs["skip_division_by_dv"] else " / m**3"),
        ):
            super().__init__(name=name, unit=unit)

        def _impl(self, **_):
            self._download_moment_to_buffer(
                attr=kwargs["attr"],
                rank=kwargs["rank"],
                skip_division_by_m0=kwargs["skip_division_by_m0"],
            )
            if not kwargs["skip_division_by_dv"]:
                self.buffer /= self.particulator.mesh.dv
            return self.buffer

    return ArbitraryMoment


ZerothMoment = make_arbitrary_moment_product(
    rank=0,
    attr="volume",
    attr_unit="m^3",
    skip_division_by_m0=True,
    skip_division_by_dv=True,
)

VolumeFirstMoment = make_arbitrary_moment_product(
    rank=1,
    attr="volume",
    attr_unit="m^3",
    skip_division_by_m0=True,
    skip_division_by_dv=True,
)

VolumeSecondMoment = make_arbitrary_moment_product(
    rank=2,
    attr="volume",
    attr_unit="m^3",
    skip_division_by_m0=True,
    skip_division_by_dv=True,
)

RadiusSixthMoment = make_arbitrary_moment_product(
    rank=6,
    attr="radius",
    attr_unit="m",
    skip_division_by_m0=True,
    skip_division_by_dv=True,
)

RadiusFirstMoment = make_arbitrary_moment_product(
    rank=1,
    attr="radius",
    attr_unit="m",
    skip_division_by_m0=True,
    skip_division_by_dv=True,
)
