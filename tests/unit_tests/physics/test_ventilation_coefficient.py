"""
tests for ventliation coefficient implmentation
"""

import numpy as np
import pytest
from PySDM import Formulae
from PySDM.physics import si


# digitized from Zografos et al. (1987) Figure A5
@pytest.mark.parametrize(
    "T, eta",
    (
        (127.23214285714285 * si.K, 0.000010242085661080074 * si.Pa * si.s),
        (287.94642857142856 * si.K, 0.000017690875232774675 * si.Pa * si.s),
        (455.35714285714283 * si.K, 0.00002458100558659218 * si.Pa * si.s),
        (703.125 * si.K, 0.000033519553072625704 * si.Pa * si.s),
        (1011.1607142857142 * si.K, 0.00004320297951582868 * si.Pa * si.s),
        (1332.5892857142856 * si.K, 0.000052327746741154564 * si.Pa * si.s),
        (1660.7142857142856 * si.K, 0.00005996275605214153 * si.Pa * si.s),
        (1962.0535714285713 * si.K, 0.0000675977653631285 * si.Pa * si.s),
        (2270.089285714286 * si.K, 0.0000750465549348231 * si.Pa * si.s),
        (2537.9464285714284 * si.K, 0.0000824953445065177 * si.Pa * si.s),
        (2812.5 * si.K, 0.00009050279329608939 * si.Pa * si.s),
    ),
)
def test_etavT_figA5(T, eta, rtol=5e-2):
    formulae = Formulae(ventilation="PruppacherRasmussen")
    eta_test = formulae.ventilation.calc_eta_air_Earth(T)
    np.testing.assert_allclose(eta_test, eta, rtol=rtol)


# digitized from Pruppacher and Rassmussen (1979) Figure 1
@pytest.mark.parametrize(
    "X, fV",
    (
        (0.1329639889196676, 1.0201096892138939),
        (0.664819944598338, 1.0201096892138939),
        (1.662049861495845, 1.2504570383912248),
        (3.988919667590028, 1.9744058500914077),
        (6.581717451523546, 2.7970749542961606),
        (9.4404432132964, 3.685557586837294),
        (11.168975069252078, 4.212065813528336),
        (13.495844875346261, 4.936014625228519),
        (16.686980609418285, 5.9561243144424125),
        (20.210526315789476, 7.074954296160877),
        (23.534626038781166, 8.12797074954296),
        (27.52354570637119, 9.345521023765995),
        (30.84764542936288, 10.39853747714808),
        (34.57063711911358, 11.583180987202924),
        (38.1606648199446, 12.702010968921389),
        (41.21883656509696, 13.656307129798902),
        (44.60941828254848, 14.742230347349176),
        (48.53185595567867, 15.926873857404022),
    ),
)
def test_fVvX_fig1(X, fV, rtol=5e-2):
    formulae = Formulae(ventilation="PruppacherRasmussen")
    fV_test = formulae.ventilation.calcfV(X)
    np.testing.assert_allclose(fV_test, fV, rtol=rtol)
