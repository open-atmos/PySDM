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

    @staticmethod
    def test_straub_p2():
        # arrange
        formulae = Formulae(fragmentation_function="Straub2010Nf")

        # act
        frag_size = formulae.fragmentation_function.p2(CW=0.666, rand=0)

        # assert
        np.testing.assert_approx_equal(frag_size, 4.3000510e-09)

    @staticmethod
    def test_straub_p3():
        # arrange
        formulae = Formulae(fragmentation_function="Straub2010Nf")

        # act
        frag_size = formulae.fragmentation_function.p3(CW=0.666, ds=0, rand=0)

        # assert
        np.testing.assert_approx_equal(frag_size, 1.3857897e-15)

    @staticmethod
    def test_straub_p4():
        # arrange
        formulae = Formulae(fragmentation_function="Straub2010Nf")

        # act
        frag_size = formulae.fragmentation_function.p4(
            CW=0.666,
            ds=0,
            v_max=0,
            Nr1=1,
            Nr2=2,
            Nr3=0,
        )

        # assert
        np.testing.assert_approx_equal(frag_size, -5.6454883153e-06)
