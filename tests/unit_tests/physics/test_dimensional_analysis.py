import pytest
import numba
from PySDM.physics.dimensional_analysis import DimensionalAnalysis
from PySDM.physics.formulae import Formulae
from PySDM.physics import constants


class TestDimensionalAnalysis:

    @staticmethod
    def test_fake_units():
        # Arrange
        sut = DimensionalAnalysis()

        # Act & Assert
        assert type(constants.D0) == float
        with sut:
            assert type(constants.D0) != float
            assert type(constants.D0.magnitude) == float
        assert type(constants.D0) == float

    @staticmethod
    @pytest.mark.skipif("numba.config.DISABLE_JIT")
    def test_fake_numba():
        # Arrange
        sut = DimensionalAnalysis()

        # Act & Assert
        assert hasattr(Formulae().saturation_vapour_pressure.pvs_Celsius, "py_func")
        with sut:
            assert not hasattr(Formulae().saturation_vapour_pressure.pvs_Celsius, "py_func")
        assert hasattr(Formulae().saturation_vapour_pressure.pvs_Celsius, "py_func")
