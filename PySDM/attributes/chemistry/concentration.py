from PySDM.attributes.impl.intensive_attribute import IntensiveAttribute


class ConcentrationImpl(IntensiveAttribute):
    def __init__(self, particles_builder, *, what):
        super().__init__(particles_builder, name="conc_"+what, base="moles_"+what)


def Concentration(what):
    def _constructor(pb):
        return ConcentrationImpl(pb, what=what)
    return _constructor


