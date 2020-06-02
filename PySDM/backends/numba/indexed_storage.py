"""
Created at 03.06.2020
"""

from .storage import Storage


class IndexedStorage(Storage):

    def __init__(self, idx: Storage, data, shape, dtype):
        super().__init__(data, shape, dtype)
        self.idx = idx

    def amax(self):
        pass

    def amin(self):
        pass

    def distance_pair(self, other, is_first_in_pair):
        pass

    def max_pair(self, other, is_first_in_pair):
        pass

    def sum_pair_body(self, other, is_first_in_pair):
        pass

    def sum_pair(self, other, is_first_in_pair):
        pass
