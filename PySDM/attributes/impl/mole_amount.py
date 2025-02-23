"""
mole amounts (extensive, base attributes)
"""

from PySDM.attributes.impl.extensive_attribute import ExtensiveAttribute


class MoleAmountImpl(ExtensiveAttribute):
    def __init__(self, builder, *, name):
        super().__init__(builder, name=name)


def make_mole_amount_factory(compound):
    def _factory(builder):
        return MoleAmountImpl(builder, name="moles_" + compound)

    return _factory
