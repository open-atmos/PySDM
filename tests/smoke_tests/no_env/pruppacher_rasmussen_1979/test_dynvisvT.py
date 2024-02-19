"""
test checking values from [Zografos et al. (1987)](doi:10.1016/0045-7825(87)90003-X) Figure A5
"""

import numpy as np
import pytest
from PySDM import Formulae
from PySDM.physics import si


# digitized from Zografos et al. (1987) Figure A5
@pytest.mark.parametrize(
    "T, η",
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
def test_ηvT_figA5(T, η, rtol=5e-2):
    formulae = Formulae(ventilation="PruppacherRasmussen")
    η_test = formulae.ventilation.calcηairEarth(T)
    np.testing.assert_allclose(η_test, η, rtol=rtol)
