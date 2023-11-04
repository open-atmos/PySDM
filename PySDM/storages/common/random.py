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
    """
    Common base class for random number generation abstraction layer

    Attributes
    ----------
    size : int
        number of random numbers to be generated
    seed : int
        seed for random number generator
    """

    size: int
    seed: int

    def __post_init__(self):
        """
        Initialise the random number generator.

        Validates the size and seed attributes.

        Returns
        -------
        None
        """
        assert isinstance(self.size, int)
        assert isinstance(self.seed, int)

    @abc.abstractmethod
    def __call__(self, storage: "Storage"):
        """
        Generates random numbers and stores them in the storage.
        """
        raise NotImplementedError()
