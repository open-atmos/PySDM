# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np

from PySDM import Formulae


class TestFragmentationFunctions:  # pylint:disable=too-few-public-methods
    @staticmethod
    def test_straub_p1():
        # arrange
        formulae = Formulae(fragmentation_function="Straub2010Nf")
        sigma1 = formulae.fragmentation_function.sigma1(CW=0.666)

        # act
        frag_size = formulae.fragmentation_function.p1(sigma1=sigma1, rand=0)

        # assert
        np.testing.assert_approx_equal(frag_size, 3.6490627e-12)
