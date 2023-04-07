# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np

from PySDM import Formulae
from PySDM.physics.constants import si


class TestFragmentationFunctions:  # pylint:disable=too-few-public-methods
    @staticmethod
    def test_straub_sigma1():
        # arrange
        formulae = Formulae(fragmentation_function="Straub2010Nf")

        # act
        params = formulae.fragmentation_function.params_sigma1(CW=30.0)

        # assert
        np.testing.assert_array_almost_equal(params, [0.467381])

    @staticmethod
    def test_straub_mu1():
        # arrange
        formulae = Formulae(fragmentation_function="Straub2010Nf")

        # act
        params = formulae.fragmentation_function.params_mu1(sigma1=0.467381)

        # assert
        np.testing.assert_array_almost_equal(params, [-7.933269])

    @staticmethod
    def test_straub_sigma2():
        # arrange
        formulae = Formulae(fragmentation_function="Straub2010Nf")

        # act
        params = formulae.fragmentation_function.params_sigma2(CW=30.0)

        # assert
        np.testing.assert_array_almost_equal(params, [0.000182])

    @staticmethod
    def test_straub_mu2():
        # arrange
        formulae = Formulae(fragmentation_function="Straub2010Nf")

        # act
        params = formulae.fragmentation_function.params_mu2(ds=0.0)

        # assert
        np.testing.assert_array_almost_equal(params, [0.00095])

    @staticmethod
    def test_straub_sigma3():
        # arrange
        formulae = Formulae(fragmentation_function="Straub2010Nf")

        # act
        params = formulae.fragmentation_function.params_sigma3(CW=30.0)

        # assert
        np.testing.assert_array_almost_equal(params, [0.000149])

    @staticmethod
    def test_straub_mu3():
        # arrange
        formulae = Formulae(fragmentation_function="Straub2010Nf")

        # act
        params = formulae.fragmentation_function.params_mu3(ds=0.18 * si.cm)

        # assert
        np.testing.assert_array_almost_equal(params, [0.00162])

    @staticmethod
    def test_ll82_pf1():
        # arrange
        formulae = Formulae(fragmentation_function="LowList1982Nf")

        # act
        params = formulae.fragmentation_function.params_f1(
            dl=0.36 * si.cm, dcoal=0.3744 * si.cm
        )
        # assert
        np.testing.assert_array_equal(
            params, [105.78851401149461, 0.36, 0.003771383856549656]
        )

    @staticmethod
    def test_ll82_pf2():
        # arrange
        formulae = Formulae(fragmentation_function="LowList1982Nf")

        # act
        params = formulae.fragmentation_function.params_f2(ds=0.18 * si.cm)

        # assert
        np.testing.assert_array_almost_equal(
            params, (31.081892267202157, 0.18, 0.01283519925273017)
        )

    @staticmethod
    def test_ll82_pf3():
        # arrange
        formulae = Formulae(fragmentation_function="LowList1982Nf")

        # act
        params = formulae.fragmentation_function.params_f3(
            ds=0.0715 * si.cm, dl=0.18 * si.cm
        )

        # assert
        np.testing.assert_array_almost_equal(
            params, (11.078017412424996, -3.4579794266811095, 0.21024917628814235)
        )

    @staticmethod
    def test_ll82_ps1():
        # arrange
        formulae = Formulae(fragmentation_function="LowList1982Nf")

        # act
        params = formulae.fragmentation_function.params_s1(
            dl=0.36 * si.cm, ds=0.18 * si.cm, dcoal=0.3744 * si.cm
        )

        # assert
        np.testing.assert_array_almost_equal(
            params, (55.710586181217394, 0.36, 0.007344262785151853)
        )

    @staticmethod
    def test_ll82_ps2():
        # arrange
        formulae = Formulae(fragmentation_function="LowList1982Nf")

        # act
        params = formulae.fragmentation_function.params_s2(
            dl=0.36 * si.cm, ds=0.18 * si.cm, St=3.705e-6 * si.J
        )

        # assert
        np.testing.assert_array_almost_equal(
            params, (13.120297517162507, -2.0082590717125437, 0.24857168491193957)
        )

    @staticmethod
    def test_ll82_pd1():
        # arrange
        formulae = Formulae(fragmentation_function="LowList1982Nf")

        # act
        params = formulae.fragmentation_function.params_d1(
            W1=2.67, dl=0.36 * si.cm, dcoal=0.3744 * si.cm, CKE=8.55e-6 * si.J
        )

        # assert
        np.testing.assert_array_almost_equal(
            params, (24.080107809942664, 0.28666015630152986, 0.016567297254868083)
        )

    @staticmethod
    def test_ll82_pd2():
        # arrange
        formulae = Formulae(fragmentation_function="LowList1982Nf")

        # act
        params = formulae.fragmentation_function.params_d2(
            ds=0.18 * si.cm, dl=0.36 * si.cm, CKE=8.55e-6 * si.J
        )

        # assert
        np.testing.assert_array_almost_equal(params, [0.0, -4.967578, -4.967578])
