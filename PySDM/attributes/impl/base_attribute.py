from .attribute import Attribute


class BaseAttribute(Attribute):

    def __init__(self, builder, name, dtype=float, size=0):
        super().__init__(builder, name=name, dtype=dtype, size=size)

    def init(self, data):
        self.data.upload(data)
        self.mark_updated()
