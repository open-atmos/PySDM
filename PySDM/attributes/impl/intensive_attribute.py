from .derived_attribute import DerivedAttribute


class IntensiveAttribute(DerivedAttribute):
    def __init__(self, builder, name: str, base: str):
        self.volume = builder.get_attribute('volume')
        self.base = builder.get_attribute(base)
        super().__init__(builder, name, dependencies=(self.volume, self.base))

    def recalculate(self):
        self.data.idx = self.volume.data.idx
        self.data.ratio(self.base.get(), self.volume.get())
