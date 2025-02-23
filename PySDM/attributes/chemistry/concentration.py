"""
concentrations (intensive, derived attributes)
"""

from PySDM.attributes.impl import IntensiveAttribute, register_attribute
from PySDM.attributes.impl.mole_amount import make_mole_amount_factory
from PySDM.dynamics.impl.chemistry_utils import AQUEOUS_COMPOUNDS


class ConcentrationImpl(IntensiveAttribute):
    def __init__(self, builder, *, what):
        super().__init__(builder, name="conc_" + what, base="moles_" + what)


def make_concentration_factory(what):
    def _factory(builder):
        return ConcentrationImpl(builder, what=what)

    return _factory


for compound in AQUEOUS_COMPOUNDS:
    register_attribute(name=f"conc_{compound}")(make_concentration_factory(compound))

    register_attribute(name=f"moles_{compound}")(make_mole_amount_factory(compound))
