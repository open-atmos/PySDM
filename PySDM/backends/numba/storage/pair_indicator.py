"""
Created at 09.11.2020
"""

from PySDM.backends.numba.impl._algorithmic_step_methods import AlgorithmicStepMethods
from PySDM.backends.numba.storage.storage import Storage


class PairIndicator:

    def __init__(self, length):
        self.indicator = Storage.empty(length, dtype=bool)
        self.length = length

    def __len__(self):
        return self.length

    def update(self, cell_start, cell_idx, cell_id):
        AlgorithmicStepMethods.find_pairs(
            cell_start, self.indicator, cell_id, cell_idx, cell_id.idx)
        self.length = len(cell_id)
