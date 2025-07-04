"""
# TODO: define Bolin number
# TODO: consider positive/negative values?
# TODO: comment on total vs. light approximation
"""

from PySDM.attributes.impl import DerivedAttribute, register_attribute
from PySDM.dynamics.isotopic_fractionation import HEAVY_ISOTOPES


class BolinNumberImpl(DerivedAttribute):
    def __init__(self, builder, *, heavy_isotope: str):
        self.moles_heavy_isotope = builder.get_attribute("moles_" + heavy_isotope)
        self.molar_mass = getattr(
            builder.particulator.formulae.constants, f"M_{heavy_isotope}"
        )
        super().__init__(
            builder,
            name="Bolin number for " + heavy_isotope,
            dependencies=(self.moles_heavy_isotope,),
        )

    def recalculate(self):
        self.particulator.bolin_number(
            output=self.data,
            molar_mass=self.molar_mass,
            moles_heavy_isotope=self.moles_heavy_isotope.data,
        )


def make_bolin_number_factory(heavy_isotope: str):
    def _factory(builder):
        return BolinNumberImpl(builder, heavy_isotope=heavy_isotope)

    return _factory


for heavy_isotope in HEAVY_ISOTOPES:
    register_attribute(name=f"Bolin number for {heavy_isotope}")(
        make_bolin_number_factory(heavy_isotope)
    )
