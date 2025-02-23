"""
storage abstraction layer facilitating pairwise operations (for use with PairwiseStorage class)
"""


def make_PairIndicator(backend):
    class PairIndicator:
        def __init__(self, length):
            self.indicator = backend.Storage.empty(length, dtype=bool)
            self.length = length

        def __len__(self):
            return self.length

        def update(self, cell_start, cell_idx, cell_id):
            backend.find_pairs(cell_start, self, cell_id, cell_idx, cell_id.idx)
            self.length = len(cell_id)

    return PairIndicator
