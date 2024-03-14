"""
resolves fractionation of water molecules across different isotopologues
requires condensation dynamic to be registered (and run beforehand)
"""

from PySDM.dynamics.condensation import Condensation

LIGHT_ISOTOPES = ("1H", "16O")
HEAVY_ISOTOPES = ("2H", "3H", "17O", "18O")


class IsotopicFractionation:
    def __init__(self, isotopes: tuple = HEAVY_ISOTOPES):
        self.isotopes = isotopes
        self.particulator = None

    def register(self, builder):
        self.particulator = builder.particulator

        try:
            ix_cond = list(builder.particulator.dynamics.keys()).index(
                Condensation.__name__
            )
        except ValueError:
            ix_cond = -1
        ix_self = list(builder.particulator.dynamics.keys()).index(
            self.__class__.__name__
        )
        if ix_cond == -1 or ix_cond > ix_self:
            raise AssertionError(
                f"{Condensation.__name__} needs to be registered to run prior to {self.__class__}"
            )

        for isotope in self.isotopes:
            builder.request_attribute(f"moles_{isotope}")

    def __call__(self):
        self.particulator.isotopic_fractionation(self.isotopes)
