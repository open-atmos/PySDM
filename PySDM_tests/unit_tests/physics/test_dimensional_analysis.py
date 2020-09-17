"""
Created at 2019
"""

import pytest
import os
from PySDM.physics.dimensional_analysis import DimensionalAnalysis, formulae
from PySDM.physics import constants


class TestDimensionalAnalysis:
    def test_fake_units(self):
        # Arrange
        sut = DimensionalAnalysis()

        # Act & Assert
        assert type(constants.D0) == float
        with sut:
            assert type(constants.D0) != float
            assert type(constants.D0.magnitude) == float
        assert type(constants.D0) == float

    @pytest.mark.skipif("'NUMBA_DISABLE_JIT' in os.environ and os.environ['NUMBA_DISABLE_JIT'] != '0'")
    def test_fake_numba(self):
        # Arrange
        sut = DimensionalAnalysis()

        # Act & Assert
        assert hasattr(formulae.pvs, "py_func")
        with sut:
            assert not hasattr(formulae.pvs, "py_func")
        assert hasattr(formulae.pvs, "py_func")
