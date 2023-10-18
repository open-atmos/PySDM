"""
per-droplet isotopic ratio of heavy-to-light isotope number concentrations
expressed vs the VSMOW reference in SI units (i.e., not per mille)
"""
from PySDM.attributes.impl.derived_attribute import DerivedAttribute


class DeltaImpl(DerivedAttribute):
    def __init__(self, builder, *, what):
        assert what[:-1].isnumeric()
        if what[-1] == "H":
            denom = "1H"
        elif what[-1] == "O":
            denom = "16O"
        else:
            raise NotImplementedError()

        super().__init__(
            builder,
            name="delta_" + what,
            dependencies=(
                builder.get_attribute("moles_" + what),
                builder.get_attribute(f"moles_{denom}"),
            ),
        )

        self.reference_ratio = getattr(self.formulae.constants, f"VSMOW_R_{what}")

    def recalculate(self):
        self.data.ratio(self.dependencies[0].get(), self.dependencies[1].get())
        self.particulator.backend.isotopic_delta(
            self.data, self.data, self.reference_ratio
        )


def make_delta_factory(what):
    def _factory(builder):
        return DeltaImpl(builder, what=what)

    return _factory
