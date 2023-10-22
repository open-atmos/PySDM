"""
unit tests for backend isotope-related routines
"""


class TestIsotopeMethods:
    @staticmethod
    def test_isotopic_fractionation(backend_class):
        # arrange
        backend = backend_class()

        # act
        backend.isotopic_fractionation()

        # assert
        # TODO #1063

    @staticmethod
    def test_isotopic_delta(backend_class):
        # arrange
        backend = backend_class()

        # act
        backend.isotopic_delta()

        # assert
        # TODO 1063
