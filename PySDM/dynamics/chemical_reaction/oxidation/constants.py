from PySDM.physics.constants import T_tri, si
from chempy import Substance


def depint(v):
    if hasattr(v, "magnitude"):
        return v.to_compact().to_base_units().magnitude
    return v


ROOM_TEMP = T_tri + 25 * si.K

M = si.mole / si.litre
gpm_u = si.gram / si.mole

K_H2O = depint(1e-14 * M * M)

DRY_RHO = 1800 * si.kg / (si.m ** 3)
DRY_FORMULA = "NH4HSO4"
DRY_SUBSTANCE = Substance.from_formula(DRY_FORMULA)
