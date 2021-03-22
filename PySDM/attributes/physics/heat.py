from PySDM.attributes.impl.extensive_attribute import ExtensiveAttribute


class Heat(ExtensiveAttribute):

    def __init__(self, builder):
        super().__init__(builder, name='heat')
