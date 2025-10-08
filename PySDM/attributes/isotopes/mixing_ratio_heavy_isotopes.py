from PySDM.attributes.impl import DerivedAttribute, register_attribute
from PySDM.dynamics.isotopic_fractionation import HEAVY_ISOTOPES


class MixingRatioImpl(DerivedAttribute):
    def __init__(self, builder, *, heavy_isotope: str):
        super().__init__(
            builder,
            name=f"mixing ratio {heavy_isotope}",
            dependencies=(self.moles_heavy_isotope,),
        )

    def recalculate(self):
        self.particulator.backend.mixing_ratio()


def make_mixing_ratio_factory(heavy_isotope: str):
    def _factory(builder):
        return MixingRatioImpl(builder, heavy_isotope=heavy_isotope)

    return _factory


for heavy_isotope in HEAVY_ISOTOPES:
    register_attribute(name=f"mixing ratio {heavy_isotope}")(
        make_mixing_ratio_factory(heavy_isotope)
    )
