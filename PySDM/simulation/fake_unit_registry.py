"""
Created at 14.11.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""


def fake(si_unit):
    return (1 * si_unit).to_base_units().magnitude


class FakeUnitRegistry:
    def __init__(self, si):

        self.dimensionless = 1

        for unit in ["centimetre", "metre", "gram", "kilogram", "mole", "joule", "kelvin", "second", "hectopascal",
                     "pascal", "litre", "micrometre"]:
            self.__setattr__(unit, fake(1 * si.__getattr__(unit)))
            self.__setattr__(unit + "s", fake(1 * si.__getattr__(unit + "s")))
