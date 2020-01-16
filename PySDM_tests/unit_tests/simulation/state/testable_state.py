"""
Created at 09.11.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from PySDM.simulation.state.state import State


class TestableState(State):

    def __getitem__(self, item: str):
        idx = self.particles.backend.to_ndarray(self._State__idx)
        all_valid = idx[:self.SD_num]
        if item == 'n':
            n = self.particles.backend.to_ndarray(self.n)
            result = n[all_valid]
        elif item == 'cell_id':
            cell_id = self.particles.backend.from_ndarray(self.cell_id)
            result = cell_id[all_valid]
        else:
            tensive = self.keys[item][0]
            attr = self.keys[item][1]
            attribute = self.particles.backend.to_ndarray(self.particles.backend.read_row(self.attributes[tensive], attr))
            result = attribute[all_valid]
        return result
