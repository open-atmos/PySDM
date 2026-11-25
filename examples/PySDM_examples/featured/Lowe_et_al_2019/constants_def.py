from PySDM.physics import constants_defaults, si

LOWE_CONSTS = {
    "sgm_org": 40 * si.mN / si.m,
    "delta_min": 0.1
    * si.nm,  # 0.2 in the paper, but 0.1 matches the paper plot fig 1c and 1d
    "MAC": 1,
    "HAC": 1,
    "c_pd": 1006 * si.joule / si.kilogram / si.kelvin,
    "g_std": 9.81 * si.metre / si.second**2,
    "Md": constants_defaults.R_str / 287.058 * si.joule / si.kelvin / si.kg,
    "Mv": constants_defaults.R_str / 461 * si.joule / si.kelvin / si.kg,
}
