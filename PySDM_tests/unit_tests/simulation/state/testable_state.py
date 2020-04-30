"""
Created at 09.11.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from PySDM.state import State


class TestableState(State):

    def __getitem__(self, item: str):
        n_sd = self.SD_num
        idx = self.particles.backend.to_ndarray(self._State__idx)
        all_valid = idx[:n_sd]
        if item == 'n':
            n = self.particles.backend.to_ndarray(self.n)
            result = n[all_valid]
        elif item == 'cell_id':
            cell_id = self.particles.backend.from_ndarray(self.cell_id)
            result = cell_id[all_valid]
        else:
            attr = self.keys[item]
            attribute = self.particles.backend.to_ndarray(self.particles.backend.read_row(self.attributes, attr))
            result = attribute[all_valid]
        return result
