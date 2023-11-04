# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest

from PySDM import formulae
from PySDM.physics import si


class TestFormulae:
    @staticmethod
    def test_c_inline():
        # arrange
        def fun(_, xxx):
            return min(
                xxx,
                2,
            )

        # act
        c_code = formulae._c_inline(
            fun, constants={"pi": 3.14}, xxx=0
        )  # pylint: disable=protected-access

        # assert
        assert ", )" not in c_code

    @staticmethod
    @pytest.mark.parametrize(
        "formulae_init_args",
        (
            {"surface_tension": "Constant"},
            {
                "surface_tension": "CompressedFilmOvadnevaite",
                "constants": {"sgm_org": 40 * si.mN / si.m, "delta_min": 0.1 * si.nm},
            },
            {
                "surface_tension": "CompressedFilmRuehl",
                "constants": {
                    "RUEHL_nu_org": 1e2 * si.cm**3 / si.mole,
                    "RUEHL_A0": 115e-20 * si.m * si.m,
                    "RUEHL_C0": 6e-7,
                    "RUEHL_m_sigma": 0.3e17 * si.J / si.m**2,
                    "RUEHL_sgm_min": 40.0 * si.mN / si.m,
                },
            },
        ),
    )
    @pytest.mark.parametrize("temp", (300.0, np.array([300, 301])))
    @pytest.mark.parametrize("v_wet", (1e-8**3.0, np.array([1e-8**3, 2e-8**3])))
    @pytest.mark.parametrize("v_dry", (1e-9**3.0, np.array([1e-9**3, 2e-9**3])))
    @pytest.mark.parametrize("f_org", (0.5, np.array([1.0, 0.5])))
    def test_trickier_formula_vectorised(formulae_init_args, temp, v_wet, v_dry, f_org):
        # arrange
        sut = formulae.Formulae(**formulae_init_args).surface_tension.sigma

        # act
        actual = sut(temp, v_wet, v_dry, f_org)

        # assert
        expected = np.empty((1 if isinstance(actual, float) else actual.size))
        for i, _ in enumerate(expected):
            expected[i] = sut(
                temp if isinstance(temp, float) else temp[i],
                v_wet if isinstance(v_wet, float) else v_wet[i],
                v_dry if isinstance(v_dry, float) else v_dry[i],
                f_org if isinstance(f_org, float) else f_org[i],
            )
        np.testing.assert_array_equal(actual, expected)
