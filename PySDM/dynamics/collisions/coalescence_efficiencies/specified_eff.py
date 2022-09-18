"""
Taking the form of Berry 1967
Cloud Droplet Growth by Collection
but with user-specified collection efficiency constants
"""

from ._parameterized import Parameterized


class SpecifiedEff(Parameterized):  # pylint: disable=too-few-public-methods
    def __init__(
        self,
        *,
        A=1,
        B=1,
        D1=-27,
        D2=1.65,
        E1=-58,
        E2=1.9,
        F1=15,
        F2=1.13,
        G1=16.7,
        G2=1,
        G3=0.004,
        Mf=4,
        Mg=8,
    ):
        super().__init__((A, B, D1, D2, E1, E2, F1, F2, G1, G2, G3, Mf, Mg))
