"""checks verifying imeplementation of `discretise_multiplicities` function"""

import pytest
import numpy as np

from PySDM.initialisation import discretise_multiplicities


class TestDiscretiseMultiplicities:
    @staticmethod
    def test_reporting_zero_multiplicity():
        """should raise an ValueError if int-casting results in zeros"""
        # arrange
        values = np.asarray([0.1], dtype=np.float64)

        # act
        with pytest.raises(ValueError) as excinfo:
            discretise_multiplicities(values)

        # assert
        assert "int-casting resulted in multiplicity of zero" in str(excinfo.value)

    @staticmethod
    def test_reporting_sum_error():
        """should err if sum of int-casted values differs a lot from the sum of floats"""
        # arrange
        values = np.asarray(
            [
                1.49,
            ],
            dtype=np.float64,
        )

        # act
        with pytest.raises(ValueError) as excinfo:
            discretise_multiplicities(values)

        # assert
        assert (
            "error in total real-droplet number due to casting multiplicities to ints"
            in str(excinfo.value)
        )

    @staticmethod
    def test_nans_converted_to_zeros():
        """checks for how NaN values are treated"""
        # arrange
        values = np.asarray(
            [
                np.nan,
                1,
            ],
            dtype=np.float64,
        )

        # act
        ints = discretise_multiplicities(values)

        # assert
        assert ints[0] == 0

    @staticmethod
    def test_all_nans_to_all_zeros():
        """checks for how NaN values are treated"""
        # arrange
        values = np.asarray(
            [
                np.nan,
                np.nan,
                np.nan,
            ],
            dtype=np.float64,
        )

        # act
        ints = discretise_multiplicities(values)

        # assert
        assert (ints == 0).all()
