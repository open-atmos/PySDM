"""
kernel modelling influence of electric field of 3000V/cm
 as in [Berry 1967](https://doi.org/10.1175/1520-0469(1967)024%3C0688:CDGBC%3E2.0.CO;2)
"""

from PySDM.dynamics.collisions.collision_kernels.impl.parameterized import Parameterized


class Electric(Parameterized):  # pylint: disable=too-few-public-methods
    def __init__(self):
        super().__init__(
            (1, 1, -7, 1.78, -20.5, 1.73, 0.26, 1.47, 1, 0.82, -0.003, 4.4, 8),
        )
