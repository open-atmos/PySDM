from PySDM.attributes.impl.base_attribute import BaseAttribute


class Multiplicities(BaseAttribute):

    def __init__(self, builder):
        super().__init__(builder, name='n', dtype=int)
