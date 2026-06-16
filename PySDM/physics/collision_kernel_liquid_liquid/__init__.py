"""
Collision kernels including
[Golovin](https://open-atmos.github.io/PySDM/PySDM/dynamics/collisions/collision_kernels/golovin.html),
[Geometric](https://open-atmos.github.io/PySDM/PySDM/dynamics/collisions/collision_kernels/geometric.html)
and other...
"""  # pylint: disable=line-too-long

from .neglect import Neglect
from .golovin import Golovin
