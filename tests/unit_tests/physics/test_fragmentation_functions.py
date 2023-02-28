# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import os

import numpy as np

from PySDM import Formulae
from PySDM.physics.constants import si

os.environ["NUMBA_DISABLE_JIT"] = "1"


class TestFragmentationFunctions:  # pylint:disable=too-few-public-methods
    # @staticmethod
    # def test_straub_p1():
    #     # arrange
    #     formulae = Formulae(fragmentation_function="Straub2010Nf")
    #     sigma1 = formulae.fragmentation_function.sigma1(CW=0.666)

    #     # act
    #     frag_size = formulae.fragmentation_function.p1(sigma1=sigma1, rand=0)

    #     # assert
    #     np.testing.assert_approx_equal(frag_size, 3.6490627e-12)

    # @staticmethod
    # def test_straub_p2():
    #     # arrange
    #     formulae = Formulae(fragmentation_function="Straub2010Nf")

    #     # act
    #     frag_size = formulae.fragmentation_function.p2(CW=0.666, rand=0)

    #     # assert
    #     np.testing.assert_approx_equal(frag_size, 4.3000510e-09)

    # @staticmethod
    # def test_straub_p3():
    #     # arrange
    #     formulae = Formulae(fragmentation_function="Straub2010Nf")

    #     # act
    #     frag_size = formulae.fragmentation_function.p3(CW=0.666, ds=0, rand=0)

    #     # assert
    #     np.testing.assert_approx_equal(frag_size, 1.3857897e-15)

    # @staticmethod
    # def test_straub_p4():
    #     # arrange
    #     formulae = Formulae(fragmentation_function="Straub2010Nf")

    #     # act
    #     frag_size = formulae.fragmentation_function.p4(
    #         CW=0.666,
    #         ds=0,
    #         v_max=0,
    #         Nr1=1,
    #         Nr2=2,
    #         Nr3=0,
    #     )

    #     # assert
    #     np.testing.assert_approx_equal(frag_size, -5.6454883153e-06)

    @staticmethod
    def test_ll82_pf1():
        # arrange
        formulae = Formulae(fragmentation_function="LowList1982Nf")

        # act
        params = formulae.fragmentation_function.params_f1(
            dl=100 * si.um, dcoal=200 * si.um
        )

        # assert
        np.testing.assert_approx_equal(len(params), 3.0)

    @staticmethod
    def test_ll82_pf2():
        # arrange
        formulae = Formulae(fragmentation_function="LowList1982Nf")

        # act
        params = formulae.fragmentation_function.params_f2(ds=100 * si.um)

        # assert
        np.testing.assert_approx_equal(len(params), 3.0)

    @staticmethod
    def test_ll82_pf3():
        # arrange
        formulae = Formulae(fragmentation_function="LowList1982Nf")

        # act
        params = formulae.fragmentation_function.params_f3(
            ds=100 * si.um, dl=200 * si.um
        )

        # assert
        np.testing.assert_approx_equal(len(params), 3.0)

    @staticmethod
    def test_ll82_ps1():
        # arrange
        formulae = Formulae(fragmentation_function="LowList1982Nf")

        # act
        params = formulae.fragmentation_function.params_s1(
            dl=100 * si.um, ds=50 * si.um, dcoal=200 * si.um
        )

        # assert
        np.testing.assert_approx_equal(len(params), 3.0)

    @staticmethod
    def test_ll82_ps2():
        # arrange
        formulae = Formulae(fragmentation_function="LowList1982Nf")

        # act
        params = formulae.fragmentation_function.params_s2(
            dl=100 * si.um, ds=50 * si.um, St=1e-6 * si.J
        )

        # assert
        np.testing.assert_approx_equal(len(params), 3.0)

    @staticmethod
    def test_ll82_pd1():
        # arrange
        formulae = Formulae(fragmentation_function="LowList1982Nf")

        # act
        params = formulae.fragmentation_function.params_d1(
            W1=1.0, dl=100 * si.um, CKE=1.0
        )

        # assert
        np.testing.assert_approx_equal(len(params), 3.0)

    @staticmethod
    def test_ll82_pd2():
        # arrange
        formulae = Formulae(fragmentation_function="LowList1982Nf")

        # act
        params = formulae.fragmentation_function.params_d2(
            ds=100 * si.um, dl=100 * si.um, CKE=200 * si.um
        )

        # assert
        np.testing.assert_approx_equal(len(params), 3.0)

    @staticmethod
    def test_erfinv():
        # arrange
        formulae = Formulae(fragmentation_function="LowList1982Nf")

        # act
        params = formulae.fragmentation_function.erfinv(0.5)

        # assert
        diff = np.abs(params - 0.476936)
        np.testing.assert_array_less(diff, 1e-3)
