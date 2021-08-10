def _fake(si_unit):
    return (1. * si_unit).to_base_units().magnitude


class FakeUnitRegistry:

    def __init__(self, si):
        self.dimensionless = 1.
        for prefix in ("nano", "micro", "milli", "centi", "", "hecto", "kilo"):
            for unit in ("metre", "gram", "hertz", "mole", "joule", "kelvin",
                         "second", "minute", "pascal", "litre", "hour", "newton"):
                self.__setattr__(prefix + unit, _fake(si.__getattr__(prefix + unit)))
                self.__setattr__(prefix + unit + "s", _fake(si.__getattr__(prefix + unit + "s")))

        for prefix in ("n", "u", "m", "c", "", "h", "k"):
            for unit in ("m", "g", "Hz", "mol", "J", "K", "s", "min", "day", "Pa", "l", "h", "bar", "N"):
                self.__setattr__(prefix + unit, _fake(si.__getattr__(prefix + unit)))
