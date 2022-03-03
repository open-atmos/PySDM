"""
common base class for random number generation abstraction layer
"""


class RandomCommon:
    def __init__(self, size: int, seed: int):
        assert isinstance(size, int)
        assert isinstance(seed, int)
        self.size = size
