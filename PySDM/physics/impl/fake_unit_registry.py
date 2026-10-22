"""logic around `PySDM.physics.impl.fake_unit_registry.FakeUnitRegistry` - PySDM mock of Pint's
[UnitRegistry](https://pint.readthedocs.io/en/stable/api/base.html#pint.UnitRegistry),
with the genuine Pint class used only within unit tests through the
`PySDM.physics.dimensional_analysis.DimensionalAnalysis` context manager
"""

from itertools import product
from pint.errors import UndefinedUnitError, OffsetUnitCalculusError


class FakeUnitRegistry:  # pylint: disable=too-few-public-methods
    def __init__(self, si):
        self.dimensionless = 1.0
        for prefix, unit, suffix in product(
            si._prefixes,
            tuple(v for vs in si._units_casei.values() for v in vs),
            si._suffixes,
        ):
            name = prefix + unit + suffix
            try:
                self.__setattr__(
                    name, (1.0 * si.__getattr__(name)).to_base_units().magnitude
                )
            except (UndefinedUnitError, OffsetUnitCalculusError):
                pass
