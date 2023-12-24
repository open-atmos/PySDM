""" common data for use in the
  [Pierchala et al. 2022](https://10.1016/j.gca.2022.01.020) example
"""

from PySDM.physics import constants_defaults as const
from PySDM.physics import si

# Krakow tap water isotopic composition from the Supplement
deltas_0_SMOW = {
    "2H": -62.01 * const.PER_MILLE,
    "18O": -8.711 * const.PER_MILLE,
    "17O": -4.58 * const.PER_MILLE,
}

TABLE_1 = {
    "Experiment I": {
        "RH": (0.404, 0.582, 0.792),
        "color": ("red", "blue", "green"),
        "T": (
            const.T0 + 19.61 * si.K,
            const.T0 + 19.54 * si.K,
            const.T0 + 18.97 * si.K,
        ),
    }
}

TABLE_2 = {
    "eps_diff": {
        "2H": 25.1 * const.PER_MILLE,
        "18O": 28.5 * const.PER_MILLE,
        "17O": 14.6 * const.PER_MILLE,
    },
    "n": {
        "2H": 0.90,
        "18O": 0.956,
        "17O": 0.958,
    },
    "eps_kin": {
        "2H": 9.5 * const.PER_MILLE,
        "18O": 11.45 * const.PER_MILLE,
        "17O": 5.88 * const.PER_MILLE,
    },
}
