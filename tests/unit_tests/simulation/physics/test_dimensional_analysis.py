from PySDM.simulation.physics.dimensional_analysis import DimensionalAnalysis
from PySDM.simulation.physics import constants


class TestDimensionalAnalysis:
    def test(self):
        # Arrange
        sut = DimensionalAnalysis()

        # Act & Assert
        assert type(constants.D0) == float
        with sut:
            assert type(constants.D0) != float
            assert type(constants.D0.magnitude) == float
        assert type(constants.D0) == float
