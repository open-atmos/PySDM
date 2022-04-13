"""
default values for constants which can be altered by providing alternative
values in a constants dictionary passed to Formulae __init__ method
"""
import numpy as np
from chempy import Substance
from scipy import constants as sci

from .constants import (  # pylint: disable=unused-import
    FOUR,
    ONE_THIRD,
    PI,
    PI_4_3,
    T0,
    THREE,
    TWO,
    TWO_THIRDS,
    M,
    si,
)

Md = (
    0.78 * Substance.from_formula("N2").mass * si.gram / si.mole
    + 0.21 * Substance.from_formula("O2").mass * si.gram / si.mole
    + 0.01 * Substance.from_formula("Ar").mass * si.gram / si.mole
)
Mv = Substance.from_formula("H2O").mass * si.gram / si.mole

R_str = sci.R * si.joule / si.kelvin / si.mole
N_A = sci.N_A / si.mole

eps = Mv / Md
Rd = R_str / Md
Rv = R_str / Mv

D0 = 2.26e-5 * si.metre**2 / si.second
D_exp = 1.81

K0 = 2.4e-2 * si.joules / si.metres / si.seconds / si.kelvins

# mass and heat accommodation coefficients
MAC = 1.0
HAC = 1.0

p1000 = 1000 * si.hectopascals
c_pd = 1005 * si.joule / si.kilogram / si.kelvin
c_pv = 1850 * si.joule / si.kilogram / si.kelvin
g_std = sci.g * si.metre / si.second**2

c_pw = 4218 * si.joule / si.kilogram / si.kelvin

Rd_over_c_pd = Rd / c_pd

ARM_C1 = 6.1094 * si.hectopascal
ARM_C2 = 17.625 * si.dimensionless
ARM_C3 = 243.04 * si.kelvin

FWC_C0 = 6.115836990e000 * si.hPa
FWC_C1 = 0.444606896e000 * si.hPa / si.K
FWC_C2 = 0.143177157e-01 * si.hPa / si.K**2
FWC_C3 = 0.264224321e-03 * si.hPa / si.K**3
FWC_C4 = 0.299291081e-05 * si.hPa / si.K**4
FWC_C5 = 0.203154182e-07 * si.hPa / si.K**5
FWC_C6 = 0.702620698e-10 * si.hPa / si.K**6
FWC_C7 = 0.379534310e-13 * si.hPa / si.K**7
FWC_C8 = -0.321582393e-15 * si.hPa / si.K**8

FWC_I0 = 6.098689930e000 * si.hPa
FWC_I1 = 0.499320233e000 * si.hPa / si.K
FWC_I2 = 0.184672631e-01 * si.hPa / si.K**2
FWC_I3 = 0.402737184e-03 * si.hPa / si.K**3
FWC_I4 = 0.565392987e-05 * si.hPa / si.K**4
FWC_I5 = 0.521693933e-07 * si.hPa / si.K**5
FWC_I6 = 0.307839583e-09 * si.hPa / si.K**6
FWC_I7 = 0.105785160e-11 * si.hPa / si.K**7
FWC_I8 = 0.161444444e-14 * si.hPa / si.K**8

L77W_A0 = 6.107799961 * si.hPa
L77W_A1 = 4.436518521e-1 * si.hPa / si.K
L77W_A2 = 1.428945805e-2 * si.hPa / si.K**2
L77W_A3 = 2.650648471e-4 * si.hPa / si.K**3
L77W_A4 = 3.031240396e-6 * si.hPa / si.K**4
L77W_A5 = 2.034080948e-8 * si.hPa / si.K**5
L77W_A6 = 6.136820929e-11 * si.hPa / si.K**6

L77I_A0 = 6.109177956 * si.hPa
L77I_A1 = 5.03469897e-1 * si.hPa / si.K
L77I_A2 = 1.886013408e-2 * si.hPa / si.K**2
L77I_A3 = 4.176223716e-4 * si.hPa / si.K**3
L77I_A4 = 5.824720280e-6 * si.hPa / si.K**4
L77I_A5 = 4.838803174e-8 * si.hPa / si.K**5
L77I_A6 = 1.838826904e-10 * si.hPa / si.K**6

rho_w = 1 * si.kilograms / si.litres
rho_i = 916.8 * si.kg / si.metres**3
pH_w = 7
sgm_w = 0.072 * si.joule / si.metre**2
nu_w = Mv / rho_w

p_tri = 611.73 * si.pascal
T_tri = 273.16 * si.kelvin
l_tri = 2.5e6 * si.joule / si.kilogram

l_l19_a = 0.167 * si.dimensionless
l_l19_b = 3.65e-4 / si.kelvin

k_l19_a = 4.2e-3 * si.joules / si.metres / si.seconds / si.kelvins
k_l19_b = 1.0456 * si.dimensionless
k_l19_c = 0.017 / si.kelvin

d_l19_a = 0.211e-4 * si.metre**2 / si.second
d_l19_b = 1.94

MK05_ICE_C1 = 1 * si.Pa
MK05_ICE_C2 = 9.550426 * si.dimensionless
MK05_ICE_C3 = 5723.265 * si.K
MK05_ICE_C4 = 3.53068 * si.dimensionless
MK05_ICE_C5 = 1 * si.K
MK05_ICE_C6 = 0.00728332 / si.K

MK05_LIQ_C1 = 1 * si.Pa
MK05_LIQ_C2 = 54.842763 * si.dimensionless
MK05_LIQ_C3 = 6763.22 * si.K
MK05_LIQ_C4 = 4.210 * si.dimensionless
MK05_LIQ_C5 = 1 * si.K
MK05_LIQ_C6 = 0.000367 / si.K
MK05_LIQ_C7 = 0.0415 / si.K
MK05_LIQ_C8 = 218.8 * si.K
MK05_LIQ_C9 = 53.878 * si.dimensionless
MK05_LIQ_C10 = 1331.22 * si.K
MK05_LIQ_C11 = 9.44523 * si.dimensionless
MK05_LIQ_C12 = 1 * si.K
MK05_LIQ_C13 = 0.014025 / si.K

# standard pressure and temperature (ICAO)
T_STP = (sci.zero_Celsius + 15) * si.kelvin
p_STP = 101325 * si.pascal
rho_STP = p_STP / Rd / T_STP

ROOM_TEMP = T_tri + 25 * si.K
H_u = M / p_STP
dT_u = si.K

sgm_org = np.nan
delta_min = np.nan

RUEHL_nu_org = np.nan
RUEHL_A0 = np.nan
RUEHL_C0 = np.nan
RUEHL_m_sigma = np.nan
RUEHL_sgm_min = np.nan

BIGG_DT_MEDIAN = np.nan

NIEMAND_A = np.nan
NIEMAND_B = np.nan

ABIFM_M = np.inf
ABIFM_C = np.inf
ABIFM_UNIT = 1 / si.cm**2 / si.s

J_HET = np.nan
