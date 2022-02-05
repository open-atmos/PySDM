# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
from PySDM import Formulae


class TestSurfaceTension:
    @staticmethod
    def test_compressed_film_ruehl_call_does_not_fail():
        # arrange
        sut = Formulae(
            surface_tension='CompressedFilmRuehl',
            constants={
                'RUEHL_nu_org': 1,
                'RUEHL_A0': 1,
                'RUEHL_C0': 1,
                'RUEHL_m_sigma': 1,
                'RUEHL_sgm_min': 1
            }
        ).surface_tension

        # act
        sut.sigma(T=0, v_wet=0, v_dry=0, f_org=0)
