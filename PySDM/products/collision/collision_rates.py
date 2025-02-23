"""
Rates of collision events and their deficit wrt expected values in case mutiplicity
 values preclude representation of resultant multiplicity in multiple collisions,
 or due to unrepresentable breakups from integer overflow
"""

from PySDM.products.impl import RateProduct, register_product


@register_product()
class CollisionRateDeficitPerGridbox(RateProduct):
    def __init__(self, name=None, unit="s^-1 kg^-1"):
        super().__init__(
            name=name, unit=unit, counter="collision_rate_deficit", dynamic="Collision"
        )


@register_product()
class CollisionRatePerGridbox(RateProduct):
    def __init__(self, name=None, unit="s^-1 kg^-1"):
        super().__init__(
            name=name, unit=unit, counter="collision_rate", dynamic="Collision"
        )


@register_product()
class CoalescenceRatePerGridbox(RateProduct):
    def __init__(self, name=None, unit="s^-1 kg^-1"):
        super().__init__(
            name=name, unit=unit, counter="coalescence_rate", dynamic="Collision"
        )


@register_product()
class BreakupRatePerGridbox(RateProduct):
    def __init__(self, name=None, unit="s^-1 kg^-1"):
        super().__init__(
            name=name, unit=unit, counter="breakup_rate", dynamic="Collision"
        )


@register_product()
class BreakupRateDeficitPerGridbox(RateProduct):
    def __init__(self, name=None, unit="s^-1 kg^-1"):
        super().__init__(
            name=name, unit=unit, counter="breakup_rate_deficit", dynamic="Collision"
        )
