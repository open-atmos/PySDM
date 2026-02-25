"""
Derived droplet attribute representing the Bolin number for heavy
water isotopologues (D, 17O, 18O substitutions).

The Bolin number is a dimensionless coefficient relating the
heavy-isotope mole tendency to the bulk liquid-water mass tendency
during phase change.
Registered for all isotopes listed in ``HEAVY_ISOTOPES``.
"""

from PySDM.attributes.impl import DerivedAttribute, register_attribute
from PySDM.dynamics.isotopic_fractionation import HEAVY_ISOTOPES


class BolinNumberImpl(DerivedAttribute):
    """Backend-evaluated Bolin number for a selected heavy isotopologue."""

    def __init__(self, builder, *, heavy_isotope: str):
        """
        Parameters
        ----------
        builder
            Attribute builder instance.
        heavy_isotope : str
            Heavy isotopologue identifier (entry of ``HEAVY_ISOTOPES``).
        """
        self.moles_heavy = builder.get_attribute(f"moles_{heavy_isotope}")
        self.moles_light = builder.get_attribute("moles light water")
        self.cell_id = builder.get_attribute("cell id")
        self.molality_in_dry_air = builder.particulator.environment[
            f"molality {heavy_isotope} in dry air"
        ]
        super().__init__(
            builder,
            name="Bolin number for " + heavy_isotope,
            dependencies=(self.moles_heavy, self.moles_light, self.cell_id),
        )

    def recalculate(self):
        """Recomputes the Bolin number using backend implementation."""
        self.particulator.backend.bolin_number(
            output=self.data,
            cell_id=self.cell_id.data,
            relative_humidity=self.particulator.environment["RH"],
            temperature=self.particulator.environment["T"],
            density_dry_air=self.particulator.environment["dry_air_density"],
            moles_light_molecule=self.moles_light.data,
            moles_heavy=self.moles_heavy.data,
            molality_in_dry_air=self.molality_in_dry_air,
        )


def make_bolin_number_factory(heavy_isotope: str):
    """Returns an attribute factory for the given heavy isotopologue."""

    def _factory(builder):
        """Instantiates ``BolinNumberImpl`` for the provided builder."""
        return BolinNumberImpl(builder, heavy_isotope=heavy_isotope)

    return _factory


for iso in HEAVY_ISOTOPES:
    register_attribute(name=f"Bolin number for {iso}")(make_bolin_number_factory(iso))
