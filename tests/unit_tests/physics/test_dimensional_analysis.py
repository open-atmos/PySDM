# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numba
import pytest

from PySDM.formulae import Formulae
from PySDM.physics import constants_defaults
from PySDM.physics.dimensional_analysis import DimensionalAnalysis

assert numba.config.DISABLE_JIT is not None  # pylint: disable=no-member


class TestDimensionalAnalysis:
    @staticmethod
    def test_fake_units():
        # Arrange
        sut = DimensionalAnalysis()

        # Act & Assert
        assert isinstance(constants_defaults.D0, float)
        with sut:
            assert not isinstance(constants_defaults.D0, float)
            assert isinstance(constants_defaults.D0.magnitude, float)
        assert isinstance(constants_defaults.D0, float)

    @staticmethod
    @pytest.mark.skipif("numba.config.DISABLE_JIT")
    def test_fake_numba():
        # Arrange
        sut = DimensionalAnalysis()

        # Act & Assert
        assert hasattr(Formulae().saturation_vapour_pressure.pvs_Celsius, "py_func")
        with sut:
            assert not hasattr(
                Formulae().saturation_vapour_pressure.pvs_Celsius, "py_func"
            )
        assert hasattr(Formulae().saturation_vapour_pressure.pvs_Celsius, "py_func")
