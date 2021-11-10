import numpy as np

J_HET = np.nan


class Constant:
    @staticmethod
    def _check():
        assert np.isfinite(J_HET)

    @staticmethod
    def j_het(a_w_ice):  # pylint: disable=unused-argument
        return J_HET
