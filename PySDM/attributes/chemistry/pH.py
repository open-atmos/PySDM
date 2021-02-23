from PySDM.attributes.intensive_attribute import IntensiveAttribute


class pH(IntensiveAttribute):
    def __init__(self, builder):
        self.volume = builder.get_attribute('volume')
        super().__init__(builder, name='pH', base='Hp')

    def recalculate(self):
        super().recalculate()
        self.data.log10()
        self.data *= -1
