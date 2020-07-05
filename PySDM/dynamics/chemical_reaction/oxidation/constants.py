"""
Created at 18.05.2020

@author: Grzegorz Łazarski
"""


from PySDM.physics.constants import T_tri, p_STP, si
from chempy import Substance


def depint(v):
    if hasattr(v, "magnitude"):
        return v.to_compact().to_base_units().magnitude
    return v


# Units
M = si.mole / si.litre
gpm_u = si.gram / si.mole
ppb = 1e-9
ppm = 1e-6
H_u = M / p_STP
dT_u = si.K
k_u = 1 / (si.s * M)
Da_u = si.m ** 2 / si.s

ROOM_TEMP = T_tri + 25 * si.K
K_H2O = 1e-14 * M * M

DRY_RHO = 1800 * si.kg / (si.m ** 3)
DRY_FORMULA = "NH4HSO4"
DRY_SUBSTANCE = Substance.from_formula(DRY_FORMULA)

# WolframAlpha
dry_air_d = (1.2750 * si.kg / si.m ** 3)
