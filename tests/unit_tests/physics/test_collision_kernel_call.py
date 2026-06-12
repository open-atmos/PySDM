"""
test for collision kernel basics
"""

import pytest
from PySDM.formulae import Formulae, _choices

from PySDM.physics import collision_kernel_liquid_liquid


class TestCollisionKernel:
    @staticmethod
    @pytest.mark.parametrize("variant", _choices(collision_kernel_liquid_liquid))
    def test_collision_kernel(variant):
        formulae = Formulae(
            collision_kernel_liquid_liquid=variant,
        )
