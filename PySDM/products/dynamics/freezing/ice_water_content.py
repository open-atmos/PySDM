from ...product import MomentProduct

class IceWaterContent(MomentProduct):

    def __init__(self):
        super().__init__(
            name='qi',
            unit='g/kg',
            description=f'Ice water mixing ratio'
        )
