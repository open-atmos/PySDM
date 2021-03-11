from PySDM.attributes.impl.intensive_attribute import IntensiveAttribute
from PySDM.physics.constants import convert_to
from PySDM.physics import si


class pH(IntensiveAttribute):
    def __init__(self, builder):
        self.volume = builder.get_attribute('volume')
        super().__init__(builder, name='pH', base='moles_H')

    def recalculate(self):
        super().recalculate()
        convert_to(self.data, si.litre**-1)
        self.data.log10()
        self.data *= -1
