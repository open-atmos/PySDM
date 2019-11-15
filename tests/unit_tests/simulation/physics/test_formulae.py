from PySDM.simulation.physics import constants
from PySDM.simulation.physics.dimensional_analysis import DimensionalAnalysis
from PySDM.simulation.physics.formulae import Formulae

import pytest

class TestFormulae:

    def test_pvs(self):
        with DimensionalAnalysis():
            # Arrange
            si = constants.si
            sut = Formulae.pvs.py_func
            T = 300 * si.kelvins

            # Act
            pvs = sut(T)

            # Assert
            assert pvs.units == si.hectopascals

    @pytest.mark.xfail
    def test_r_cr(self):
        with DimensionalAnalysis():
            # Arrange
            si = constants.si
            sut = Formulae.r_cr.py_func

            kp = .5
            rd = .1 * si.micrometre
            T = 300 * si.kelvins

            # Act
            r_cr = sut(kp, rd, T)

            # Assert
            assert r_cr.units == si.micrometres

