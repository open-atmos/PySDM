"""checks for SI units conversions in FakeUnitRegistry"""

import pytest

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

    @staticmethod
    def test_atm():
        """a check for the 'atm' unit"""
        assert si.atm == 101325 * si.Pa

    @staticmethod
    def test_mmHg():
        """a check for the 'mmHg' unit"""
        assert si.mmHg == pytest.approx(133.322387415 * si.Pa)

    @staticmethod
    def test_psi():
        """a check for the 'psi' unit"""
        assert si.psi == pytest.approx(6894.757293168 * si.Pa)

    @staticmethod
    def test_inHg():
        """a check for the 'inHg' unit"""
        assert si.inHg == pytest.approx(25.4 * si.mmHg)

    @staticmethod
    def test_inch():
        """a check for the 'inch' unit"""
        assert si.inch == pytest.approx(2.54 * si.cm)

    @staticmethod
    def test_cc():
        """a check for the 'cc' unit"""
        assert si.cc == si.cm**3

    @staticmethod
    def test_percent():
        """a check for the 'percent' unit"""
        assert 100 * si.percent == 1

    @staticmethod
    def test_permille():
        """a check for the 'permille' unit"""
        assert 1000 * si.permille == 1
