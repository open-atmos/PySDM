"""
unit tests for backend isotope-related routines
"""

import numpy as np


class TestIsotopeMethods:
    @staticmethod
    def test_isotopic_fractionation(backend_instance):
        # arrange
        backend = backend_instance

        # act
        backend.isotopic_fractionation()

    @staticmethod
    def test_isotopic_delta(backend_instance):
        # arrange
        backend = backend_instance
        arr2storage = backend.Storage.from_ndarray
        n_sd = 10
        output = arr2storage(np.empty(n_sd))
        ratio = arr2storage(np.zeros(n_sd))

        # act
        backend.isotopic_delta(output=output, ratio=ratio, reference_ratio=0.0001)

        # assert
        assert (output.to_ndarray() == -1).all()
