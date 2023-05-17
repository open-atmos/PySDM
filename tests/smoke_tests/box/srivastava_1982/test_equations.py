# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
from PySDM_examples.Srivastava_1982.equations import Equations

from PySDM.physics import constants_defaults
from PySDM.physics.dimensional_analysis import DimensionalAnalysis


class TestEquations:
    def test_eq10(self):
        with DimensionalAnalysis():
            # arrange
            si = constants_defaults.si
            frag_mass = 1 * si.kg
            eqs = Equations(alpha=1 / si.s, c=1 / si.s, M=1 * si.kg / frag_mass)

            # act
            m_e = eqs.eq10(m0=0.1 * si.kg / frag_mass, tau=eqs.tau(1 * si.s))

            # assert
            assert m_e.check("[]")

    def test_eq12(self):
        with DimensionalAnalysis():
            # arrange
            si = constants_defaults.si
            frag_mass = 1 * si.kg
            eqs = Equations(alpha=1 / si.s, c=1 / si.s, M=1 * si.kg / frag_mass)

            # act
            m_e = eqs.eq12()

            # assert
            assert m_e.check("[]")

    def test_eq13(self):
        with DimensionalAnalysis():
            # arrange
            si = constants_defaults.si
            frag_mass = 1 * si.kg
            eqs = Equations(beta=1 / si.s, c=1 / si.s, M=1 * si.kg / frag_mass)

            # act
            m_e = eqs.eq13(m0=0.1 * si.kg / frag_mass, tau=eqs.tau(1 * si.s))

            # assert
            assert m_e.check("[]")

    def test_eq14(self):
        with DimensionalAnalysis():
            # arrange
            si = constants_defaults.si
            eqs = Equations(beta_star=1 * si.dimensionless)

            # act
            m_e = eqs.eq14()

            # assert
            assert m_e.check("[]")
