from .attribute import Attribute


class DummyAttribute(Attribute):
    def __init__(self, builder):
        super().__init__(builder, '')

    def get(self):
        return self.data
