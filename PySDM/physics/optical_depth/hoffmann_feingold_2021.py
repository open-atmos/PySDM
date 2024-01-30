"""
[Hoffmann and Feingold 2021]()
Eq. 3 and last paragraph of section 3a
"""

from PySDM import si


class HoffmannFeingold2021:  # pylint: disable=too-few-public-methods
    def __init__(self, _):
        pass

    @staticmethod
    def tau(
        const,
    ):
        return  # integral(Q_e(r, 500*si.nm) * const.pi * r**2 * n(r, z) dr dz)
