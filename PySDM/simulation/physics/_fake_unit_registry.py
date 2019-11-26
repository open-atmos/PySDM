"""
Created at 14.11.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""


def fake(si_unit):
    return (1. * si_unit).to_base_units().magnitude


class FakeUnitRegistry:
    def __init__(self, si):

        self.dimensionless = 1.
        for prefix in ["nano", "micro", "milli", "centi", "", "hecto", "kilo"]:
            for unit in ["metre", "gram", "gram", "mole", "joule", "kelvin", "second", "pascal", "litre"]:
                self.__setattr__(prefix+unit, fake(si.__getattr__(prefix+unit)))
                self.__setattr__(prefix+unit + "s", fake(si.__getattr__(prefix+unit + "s")))

