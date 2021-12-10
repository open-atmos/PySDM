from PySDM.attributes.impl.cell_attribute import CellAttribute


class CellID(CellAttribute):

    def __init__(self, builder):
        super().__init__(builder, name='cell id', dtype=int)

    def recalculate(self):
        pass