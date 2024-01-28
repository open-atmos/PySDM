"""
concentrations (intensive, derived attributes)
"""

from PySDM.attributes.impl.intensive_attribute import IntensiveAttribute


class ConcentrationImpl(IntensiveAttribute):
    def __init__(self, builder, *, what):
        super().__init__(builder, name="conc_" + what, base="moles_" + what)


def make_concentration_factory(what):
    def _factory(builder):
        return ConcentrationImpl(builder, what=what)

    return _factory
