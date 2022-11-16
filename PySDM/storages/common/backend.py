from typing import Protocol


class Backend(Protocol):
    ...


class IndexBackend(Backend):
    @staticmethod
    def identity_index(idx) -> None:
        ...

    @staticmethod
    def shuffle_global(idx, length, u01) -> None:
        ...

    @staticmethod
    def shuffle_local(idx, u01, cell_start) -> None:
        ...

    @staticmethod
    def sort_by_key(idx, attr) -> None:
        ...

    @staticmethod
    def remove_zero_n_or_flagged(multiplicity, idx, length) -> int:
        ...


class PairBackend(Backend):
    @staticmethod
    def distance_pair(data_out, data_in, is_first_in_pair, idx) -> None:
        ...

    @staticmethod
    def max_pair(data_out, data_in, is_first_in_pair, idx) -> None:
        ...

    @staticmethod
    def min_pair(data_out, data_in, is_first_in_pair, idx) -> None:
        ...

    @staticmethod
    def sort_pair(data_out, data_in, is_first_in_pair, idx) -> None:
        ...

    @staticmethod
    def sum_pair(data_out, data_in, is_first_in_pair, idx) -> None:
        ...

    @staticmethod
    def multiply_pair(data_out, data_in, is_first_in_pair, idx) -> None:
        ...
