""" logic around `PySDM.physics.impl.fake_unit_registry.FakeUnitRegistry` - PySDM mock of Pint's
 [UnitRegistry](https://pint.readthedocs.io/en/stable/developers_reference.html#pint.UnitRegistry),
 with the genuine Pint class used only within unit tests through the
 `PySDM.physics.dimensional_analysis.DimensionalAnalysis` context manager
"""


def _fake(si_unit):
    return (1.0 * si_unit).to_base_units().magnitude


class FakeUnitRegistry:  # pylint: disable=too-few-public-methods
    def __init__(self, si):
        self.dimensionless = 1.0
        for prefix in ("nano", "micro", "milli", "centi", "", "hecto", "kilo"):
            for unit in (
                "bar",
                "metre",
                "gram",
                "hertz",
                "mole",
                "joule",
                "kelvin",
                "second",
                "minute",
                "pascal",
                "litre",
                "hour",
                "newton",
                "watt",
            ):
                self.__setattr__(prefix + unit, _fake(si.__getattr__(prefix + unit)))
                self.__setattr__(
                    prefix + unit + "s", _fake(si.__getattr__(prefix + unit + "s"))
                )

        for prefix in ("n", "u", "m", "c", "", "h", "k"):
            for unit in (
                "b",
                "m",
                "g",
                "Hz",
                "mol",
                "J",
                "K",
                "s",
                "min",
                "day",
                "Pa",
                "l",
                "h",
                "bar",  # note: "b" is barn !!!
                "N",
                "W",
            ):
                self.__setattr__(prefix + unit, _fake(si.__getattr__(prefix + unit)))
