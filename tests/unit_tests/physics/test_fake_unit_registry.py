"""checks for SI units conversions in FakeUnitRegistry"""

from PySDM.physics import si


class TestFakeUnitRegistry:
    @staticmethod
    def test_d():
        """a check for the 'd' prefix"""
        assert 44 * si.dm == 440 * si.cm

    @staticmethod
    def test_deci():
        """a check for the 'deci' prefix"""
        assert 44 * si.decimetre == 440 * si.centimetre
