
class TestAqueousChemistry:
    @staticmethod
    def test_oxidize():
        pass
        # TODO!

    @staticmethod
    def test_henry():
        from chempy.henry import Henry
        kH_O2 = Henry(1.2e-3, 1800, ref='carpenter_1966')

        from PySDM.dynamics.aqueous_chemistry.support import HENRY_CONST

