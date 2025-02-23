"""
common base class for random number generation abstraction layer
"""


class RandomCommon:  # pylint: disable=too-few-public-methods
    def __init__(self, size: int, seed: int):
        assert isinstance(size, int)
        assert isinstance(seed, int)
        self.size = size
