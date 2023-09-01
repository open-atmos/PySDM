"""
concentration of particles within a grid cell (either per-volume of per-mass-of-dry air),
 activated, unactivated or both
"""

from PySDM.products.impl.activation_filtered_product import _ActivationFilteredProduct
from PySDM.products.impl.concentration_product import ConcentrationProduct


class ActivatedParticleConcentration(ConcentrationProduct, _ActivationFilteredProduct):
    # pylint: disable=too-many-arguments
    def __init__(
        self,
        *,
        count_unactivated: bool,
        count_activated: bool,
        specific=False,
        stp=False,
        name=None,
        unit="m^-3",
    ):
        ConcentrationProduct.__init__(
            self, name=name, unit=unit, specific=specific, stp=stp
        )
        _ActivationFilteredProduct.__init__(
            self, count_activated=count_activated, count_unactivated=count_unactivated
        )

    def register(self, builder):
        for base_class in (_ActivationFilteredProduct, ConcentrationProduct):
            base_class.register(self, builder)

    def _impl(self, **kwargs):
        _ActivationFilteredProduct.impl(self, attr="volume", rank=0)
        return ConcentrationProduct._impl(self, **kwargs)


class ActivatedParticleSpecificConcentration(ActivatedParticleConcentration):
    def __init__(self, count_unactivated, count_activated, name=None, unit="kg^-1"):
        super().__init__(
            count_unactivated=count_unactivated,
            count_activated=count_activated,
            specific=True,
            name=name,
            unit=unit,
        )
