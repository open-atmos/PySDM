from PySDM.products.impl.rate_product import RateProduct


class CollisionRateDeficitPerGridbox(RateProduct):
    def __init__(self, name=None, unit='s^-1'):
        super().__init__(name=name, unit=unit,
                         counter='collision_rate_deficit', dynamic='Coalescence')


class CollisionRatePerGridbox(RateProduct):
    def __init__(self, name=None, unit='s^-1'):
        super().__init__(name=name, unit=unit,
                         counter='collision_rate', dynamic='Coalescence')
