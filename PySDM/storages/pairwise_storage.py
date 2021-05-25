def make_PairwiseStorage(backend):

    class PairwiseStorage(backend.Storage):

        def __init__(self, data, shape, dtype):
            super().__init__(data, shape, dtype)

        @staticmethod
        def empty(shape, dtype):
            result = PairwiseStorage(*backend.Storage._get_empty_data(shape, dtype))
            return result

        @staticmethod
        def from_ndarray(array):
            result = PairwiseStorage(*backend.Storage._get_data_from_ndarray(array))
            return result

        def distance(self, other, is_first_in_pair):
            backend.distance_pair(self, other, is_first_in_pair, other.idx)

        def max(self, other, is_first_in_pair):
            backend.max_pair(self, other, is_first_in_pair, other.idx)

        def sort(self, other, is_first_in_pair):
            backend.sort_pair(self, other, is_first_in_pair, other.idx)

        def sum(self, other, is_first_in_pair):
            backend.sum_pair(self, other, is_first_in_pair, other.idx)

    return PairwiseStorage
