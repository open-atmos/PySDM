"""
Created at 09.11.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from PySDM.simulation.state.state import State


class TestableState(State):

    def __getitem__(self, item: str):
        idx = self.backend.to_ndarray(self.idx)
        all_valid = idx[:self.SD_num]
        if item == 'n':
            n = self.backend.to_ndarray(self.n)
            result = n[all_valid]
        elif item == 'cell_id':
            cell_id = self.backend.from_ndarray(self.cell_id)
            result = cell_id[all_valid]
        else:
            tensive = self.keys[item][0]
            attr = self.keys[item][1]
            attribute = self.backend.to_ndarray(self.backend.read_row(self.attributes[tensive], attr))
            result = attribute[all_valid]
        return result
