"""
# TODO: define Bolin number
# TODO: consider positive/negative values?
# TODO: comment on total vs. light approximation
"""

from PySDM.attributes.impl import DerivedAttribute, register_attribute
from PySDM.dynamics.isotopic_fractionation import HEAVY_ISOTOPES


class BolinNumberImpl(DerivedAttribute):
    def __init__(self, builder, *, heavy_isotope: str):
        self.moles_heavy = builder.get_attribute(f"moles_{heavy_isotope}")
        self.delta_heavy = builder.get_attribute(f"delta_{heavy_isotope}")
        # self.molar_mass = getattr(
        #     builder.particulator.formulae.constants, f"M_{heavy_isotope}"
        # )
        super().__init__(
            builder,
            name="Bolin number for " + heavy_isotope,
            dependencies=(
                self.moles_heavy,
                self.delta_heavy,
            ),
        )

    def recalculate(self):
        self.particulator.backend.bolin_number(
            output=self.data,
            cell_id=self.particulator.attributes["cell id"],
            relative_humidity=self.particulator.environment["RH"],
            temperature=self.particulator.environment["T"],
            moles_light_water=self.particulator.attributes["moles light water"],
            moles_heavy=self.moles_heavy.data,
            delta_heavy=self.delta_heavy.data,
        )


def make_bolin_number_factory(heavy_isotope: str):
    def _factory(builder):
        return BolinNumberImpl(builder, heavy_isotope=heavy_isotope)

    return _factory


for heavy_isotope in HEAVY_ISOTOPES:
    register_attribute(name=f"Bolin number for {heavy_isotope}")(
        make_bolin_number_factory(heavy_isotope)
    )
