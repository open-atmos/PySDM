""" """

from PySDM.attributes.impl import DerivedAttribute, register_attribute
from PySDM.dynamics.isotopic_fractionation import HEAVY_ISOTOPES


class BolinNumberImpl(DerivedAttribute):
    def __init__(self, builder, *, heavy_isotope: str):
        self.moles_heavy = builder.get_attribute(f"moles_{heavy_isotope}")
        self.molar_mixing_ratio = builder.particulator.environment[
            f"molar mixing ratio {heavy_isotope}"
        ]
        super().__init__(
            builder,
            name="Bolin number for " + heavy_isotope,
            dependencies=(self.moles_heavy,),
        )

    def recalculate(self):
        self.particulator.backend.bolin_number(
            output=self.data,
            cell_id=self.particulator.attributes["cell id"],
            relative_humidity=self.particulator.environment["RH"],
            temperature=self.particulator.environment["T"],
            density_dry_air=self.particulator.environment["dry_air_density"],
            moles_light_molecule=self.particulator.attributes["moles light water"],
            moles_heavy=self.moles_heavy.data,
            molar_mixing_ratio=self.molar_mixing_ratio.data,
        )


def make_bolin_number_factory(heavy_isotope: str):
    def _factory(builder):
        return BolinNumberImpl(builder, heavy_isotope=heavy_isotope)

    return _factory


for heavy_isotope in HEAVY_ISOTOPES:
    register_attribute(name=f"Bolin number for {heavy_isotope}")(
        make_bolin_number_factory(heavy_isotope)
    )
