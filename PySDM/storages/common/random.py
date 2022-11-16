"""
common base class for random number generation abstraction layer
"""
import abc
import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PySDM.storages.common.storage import Storage


@dataclasses.dataclass
class Random(abc.ABC):
    size: int
    seed: int

    def __post_init__(self):
        assert isinstance(self.size, int)
        assert isinstance(self.seed, int)

    @abc.abstractmethod
    def __call__(self, storage: "Storage"):
        raise NotImplementedError()
