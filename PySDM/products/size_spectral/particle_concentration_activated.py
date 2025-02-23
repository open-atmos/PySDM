"""
concentration of particles within a grid cell (either per-volume of per-mass-of-dry air),
 activated, unactivated or both
"""

from PySDM.products.impl import (
    ActivationFilteredProduct,
    ConcentrationProduct,
    register_product,
)


@register_product()
class ActivatedParticleConcentration(ConcentrationProduct, ActivationFilteredProduct):
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
        ActivationFilteredProduct.__init__(
            self, count_activated=count_activated, count_unactivated=count_unactivated
        )

    def register(self, builder):
        for base_class in (ActivationFilteredProduct, ConcentrationProduct):
            base_class.register(self, builder)

    def _impl(self, **kwargs):
        ActivationFilteredProduct.impl(self, attr="volume", rank=0)
        return ConcentrationProduct._impl(self, **kwargs)


@register_product()
class ActivatedParticleSpecificConcentration(ActivatedParticleConcentration):
    def __init__(self, count_unactivated, count_activated, name=None, unit="kg^-1"):
        super().__init__(
            count_unactivated=count_unactivated,
            count_activated=count_activated,
            specific=True,
            name=name,
            unit=unit,
        )
