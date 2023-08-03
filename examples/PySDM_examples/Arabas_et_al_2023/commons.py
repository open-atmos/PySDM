""" ABIFM and INAS parameters and other constants """

from PySDM import Formulae
from PySDM.physics import si

FREEZING_CONSTANTS = {
    "dust": {
        "NIEMAND_A": -0.517,
        "NIEMAND_B": 8.934,
        "ABIFM_M": 22.62,
        "ABIFM_C": -1.35,
    },
    "illite": {"ABIFM_M": 54.48, "ABIFM_C": -10.67},
}

COOLING_RATES = (3.75 * si.K / si.min, 0.75 * si.K / si.min, 0.15 * si.K / si.min)

BEST_FIT_LN_S_GEOM = 0.25

LOGNORMAL_MODE_SURF_A = Formulae().trivia.sphere_surface(diameter=0.74 * si.um)
LOGNORMAL_SGM_G = 2.55

TEMP_RANGE = (250 * si.K, 230 * si.K)
