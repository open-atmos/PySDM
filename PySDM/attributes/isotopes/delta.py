"""
per-droplet isotopic ratio of heavy-to-light isotope number concentrations
expressed vs the VSMOW reference in SI units (i.e., not per mille)
"""

from PySDM.attributes.impl import DerivedAttribute, register_attribute
from PySDM.dynamics.isotopic_fractionation import HEAVY_ISOTOPES


class DeltaImpl(DerivedAttribute):
    def __init__(self, builder, *, heavy_isotope):
        assert heavy_isotope[:-1].isnumeric()
        if heavy_isotope[-1] == "H":
            light_isotope = "1H"
        elif heavy_isotope[-1] == "O":
            light_isotope = "16O"
        else:
            raise NotImplementedError()

        self.heavy_isotope = builder.get_attribute("moles_" + heavy_isotope)
        self.light_isotope = builder.get_attribute(f"moles_{light_isotope}")
        super().__init__(
            builder,
            name="delta_" + heavy_isotope,
            dependencies=(
                self.heavy_isotope,
                self.light_isotope,
            ),
        )

        self.reference_ratio = getattr(
            self.formulae.constants, f"VSMOW_R_{heavy_isotope}"
        )

    def recalculate(self):
        self.data.ratio(self.heavy_isotope.get(), self.light_isotope.get())
        self.particulator.backend.isotopic_delta(
            self.data, self.data, self.reference_ratio
        )


def make_delta_factory(what):
    def _factory(builder):
        return DeltaImpl(builder, heavy_isotope=what)

    return _factory


for isotope in HEAVY_ISOTOPES:
    register_attribute(name=f"delta_{isotope}")(make_delta_factory(isotope))
