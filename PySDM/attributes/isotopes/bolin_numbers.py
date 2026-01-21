"""
Bolin number droplet attribute for heavy isotopes of water atoms
"""

# pylint: disable=missing-class-docstring, missing-function-docstring
from PySDM.attributes.impl import DerivedAttribute, register_attribute
from PySDM.dynamics.isotopic_fractionation import HEAVY_ISOTOPES


class BolinNumberImpl(DerivedAttribute):
    def __init__(self, builder, *, heavy_isotope: str):
        self.moles_heavy = builder.get_attribute(f"moles_{heavy_isotope}")
        self.moles_light = builder.get_attribute("moles light water")
        self.molality_in_dry_air = builder.particulator.environment[
            f"molality {heavy_isotope} in dry air"
        ]
        super().__init__(
            builder,
            name="Bolin number for " + heavy_isotope,
            dependencies=(self.moles_heavy, self.moles_light),
        )

    def recalculate(self):
        self.particulator.backend.bolin_number(
            output=self.data,
            cell_id=self.particulator.attributes["cell id"],
            relative_humidity=self.particulator.environment["RH"],
            temperature=self.particulator.environment["T"],
            density_dry_air=self.particulator.environment["dry_air_density"],
            moles_light_molecule=self.moles_light.data,
            moles_heavy=self.moles_heavy.data,
            molality_in_dry_air=self.molality_in_dry_air.data,
        )


def make_bolin_number_factory(heavy_isotope: str):
    def _factory(builder):
        return BolinNumberImpl(builder, heavy_isotope=heavy_isotope)

    return _factory


for iso in HEAVY_ISOTOPES:
    register_attribute(name=f"Bolin number for {iso}")(make_bolin_number_factory(iso))
