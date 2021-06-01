from PySDM.physics import constants as const
from numpy import power


# PROPERTIES OF AIR
# A Manual for Use in Biophysical Ecology
# Fourth Edition - 2010
# page 22
class TracyWelchPorter:
    @staticmethod
    def D(T, p):
        return const.D0 * power(T / const.T0, const.D_exp) * (const.p1000 / p)
