"""
mole amounts (extensive, base attributes)
"""

from PySDM.attributes.impl.extensive_attribute import ExtensiveAttribute


class MoleAmountImpl(ExtensiveAttribute):
    def __init__(self, particulator, *, name):
        super().__init__(particulator, name=name)


def make_mole_amount_factory(compound):
    def _factory(particulator):
        return MoleAmountImpl(particulator, name="moles_" + compound)

    return _factory
