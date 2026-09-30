"""
[Berry 1967](https://doi.org/10.1175/1520-0469(1967)024%3C0688:CDGBC%3E2.0.CO;2)
Cloud Droplet Growth by Collection
"""

from ._parameterized import Parameterized


class Berry1967(Parameterized):  # pylint: disable=too-few-public-methods
    def __init__(self):
        super().__init__((1, 1, -27, 1.65, -58, 1.9, 15, 1.13, 16.7, 1, 0.004, 4, 8))
