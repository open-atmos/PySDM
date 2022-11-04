"""
specialised storage equipped with particle pair-handling methods
"""


def make_PairwiseStorage(backend):
    class PairwiseStorage(backend.Storage):
        @staticmethod
        def empty(shape, dtype):
            result = PairwiseStorage(backend.Storage._get_empty_data(shape, dtype))
            return result

        @staticmethod
        def from_ndarray(array):
            result = PairwiseStorage(backend.Storage._get_data_from_ndarray(array))
            return result

        def distance(self, other, is_first_in_pair):
            backend.distance_pair(self, other, is_first_in_pair, other.idx)

        def max(self, other, is_first_in_pair):
            backend.max_pair(self, other, is_first_in_pair, other.idx)

        def min(self, other, is_first_in_pair):
            backend.min_pair(self, other, is_first_in_pair, other.idx)

        def sort(self, other, is_first_in_pair):
            backend.sort_pair(self, other, is_first_in_pair, other.idx)

        def sum(self, other, is_first_in_pair):
            """
            Sums values from `other` for each pair (e.g., drop radius for coalescence kernels).
            """
            backend.sum_pair(self, other, is_first_in_pair, other.idx)

        def multiply(self, other, is_first_in_pair):
            backend.multiply_pair(self, other, is_first_in_pair, other.idx)

    return PairwiseStorage
