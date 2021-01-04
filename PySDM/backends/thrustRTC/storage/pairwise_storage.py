"""
Created at 09.11.2020
"""

from ..impl._algorithmic_step_methods import AlgorithmicStepMethods
from .storage import Storage


class PairwiseStorage(Storage):

    def __init__(self, data, shape, dtype):
        super().__init__(data, shape, dtype)

    @staticmethod
    def empty(shape, dtype):
        result = PairwiseStorage(*Storage._get_empty_data(shape, dtype))
        return result

    @staticmethod
    def from_ndarray(array):
        result = PairwiseStorage(*Storage._get_data_from_ndarray(array))
        return result

    def distance(self, other, is_first_in_pair):
        AlgorithmicStepMethods.distance_pair(
            self.data,
            other.data,
            is_first_in_pair.indicator.data,
            other.idx.data,
            len(other)
        )

    def max(self, other, is_first_in_pair):
        AlgorithmicStepMethods.max_pair(
            self.data,
            other.data,
            is_first_in_pair.indicator.data,
            other.idx.data,
            len(other)
        )

    def sort(self, other, is_first_in_pair):
        AlgorithmicStepMethods.sort_pair(
            self.data,
            other.data,
            is_first_in_pair.indicator.data,
            other.idx.data,
            len(other)
        )

    def sum(self, other, is_first_in_pair):
        AlgorithmicStepMethods.sum_pair(
            self.data,
            other.data,
            is_first_in_pair.indicator.data,
            other.idx.data,
            len(other)
        )
