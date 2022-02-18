"""
factory for arbitrary-moment product instances
"""
from PySDM.products.impl.moment_product import MomentProduct


def make_arbitrary_moment_product(**kwargs):
    for arg in ('rank', 'name', 'unit', 'attr', 'attr_unit'):
        assert arg in kwargs.keys()

    class ArbitraryMoment(MomentProduct):
        def __init__(self, name=None, unit=f"({kwargs['attr_unit']})**{kwargs['rank']}"):
            super().__init__(name=name, unit=unit)
            self.attr = kwargs['attr']
            self.rank = kwargs['rank']

        def _impl(self, **kwargs):
            self._download_moment_to_buffer(attr=self.attr, rank=self.rank)
            return self.buffer

    return ArbitraryMoment(name=kwargs['name'], unit=kwargs['unit'])


VolumeFirstMoment = make_arbitrary_moment_product(
    rank=1, name=None, unit='m^3', attr='volume', attr_unit='m^3'
)

RadiusSixthMoment = make_arbitrary_moment_product(
    rank=6, name=None, unit='m^6', attr='radius', attr_unit='m'
)

RadiusFirstMomentInMicrometres = make_arbitrary_moment_product(
    rank=1, name=None, unit='um', attr='radius', attr_unit='m'
)
