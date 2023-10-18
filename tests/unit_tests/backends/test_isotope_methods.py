"""
unit tests for backend isotope-related routines
"""


class TestIsotopeMethods:
    @staticmethod
    def test_(backend_class):
        # arrange
        backend = backend_class()

        # act
        backend.isotopic_fractionation()

        # assert
