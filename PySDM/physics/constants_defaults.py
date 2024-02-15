"""
Default values for constants which can be altered by providing alternative
  values in a constants dictionary passed to Formulae __init__ method.
Unless, there is a very specific and sound reason, everything here should
  be provided in SI units.
"""

import numpy as np
from chempy import Substance
from scipy import constants as sci

from .constants import (  # pylint: disable=unused-import
    FOUR,
    ONE_THIRD,
    PER_CENT,
    PER_MEG,
    PER_MILLE,
    PI,
    PI_4_3,
    PPM,
    T0,
    THREE,
    ONE,
    TWO,
    TWO_THIRDS,
    M,
    si,
)
from .trivia import Trivia

# https://physics.nist.gov/cgi-bin/Star/compos.pl?matno=104
Md = (
    0.755267 * Substance.from_formula("N2").mass * si.gram / si.mole
    + 0.231781 * Substance.from_formula("O2").mass * si.gram / si.mole
    + 0.012827 * Substance.from_formula("Ar").mass * si.gram / si.mole
    + 0.000124 * Substance.from_formula("C").mass * si.gram / si.mole
)

# https://web.archive.org/web/20200729203147/https://nucleus.iaea.org/rpst/documents/VSMOW_SLAP.pdf
VSMOW_R_2H = 155.76 * PPM
VSMOW_R_3H = 1.85e-11 * PPM
VSMOW_R_18O = 2005.20 * PPM
VSMOW_R_17O = 379.9 * PPM

# https://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl?ele=H
M_1H = 1.00782503224 * si.g / si.mole
M_2H = 2.01410177812 * si.g / si.mole
M_3H = 3.01604927792 * si.g / si.mole

# https://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl?ele=O
M_16O = 15.99491461957 * si.g / si.mole
M_17O = 16.99913175651 * si.g / si.mole
M_18O = 17.99915961287 * si.g / si.mole

Mv = (
    1
    - Trivia.mixing_ratio_to_specific_content(
        VSMOW_R_2H / 2 + VSMOW_R_3H / 2 + VSMOW_R_17O + VSMOW_R_18O
    )
) * (
    (M_1H * 2 + M_16O)
    + (M_2H + M_1H + M_16O) * VSMOW_R_2H
    + (M_3H + M_1H + M_16O) * VSMOW_R_3H
    + (M_1H * 2 + M_17O) * VSMOW_R_17O
    + (M_1H * 2 + M_18O) * VSMOW_R_18O
)

R_str = sci.R * si.joule / si.kelvin / si.mole
N_A = sci.N_A / si.mole

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

p_tri = 611.73 * si.pascal
T_tri = 273.16 * si.kelvin
l_tri = 2.5e6 * si.joule / si.kilogram

# Seinfeld and Pandis, Appendix 16.1, 16A.2
# default constant values according to Lowe et al (2019), from ICPM code
l_l19_a = 0.167 * si.dimensionless
l_l19_b = 3.65e-4 / si.kelvin

# Thermal diffusivity constants from Lowe et al (2019)
k_l19_a = 4.2e-3 * si.joules / si.metres / si.seconds / si.kelvins
k_l19_b = 1.0456 * si.dimensionless
k_l19_c = 0.017 / si.kelvin

# Delta v for diffusivity in Pruppacher & Klett eq. 13-14
dv_pk05 = 0.0 * si.metres

# Seinfeld & Pandis eq. 15.65
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

ROOM_TEMP = T_tri + 25 * si.K
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

STRAUB_E_D1 = 0.04 * si.cm
STRAUB_MU2 = 0.095 * si.cm

VEDDER_1987_b = 89 / 880
VEDDER_1987_A = 993 / 880 / 3 / VEDDER_1987_b

""" [eq. 5 in Merlivat and Nief (1967)](https://doi.org/10.3402/tellusa.v19i1.9756) """
MERLIVAT_NIEF_1967_ALPHA_L_2H_T2 = 15013 * si.K**2
MERLIVAT_NIEF_1967_ALPHA_L_2H_T1 = 0 * si.K
MERLIVAT_NIEF_1967_ALPHA_L_2H_T0 = -0.1

MERLIVAT_NIEF_1967_ALPHA_I_2H_T2 = 16289 * si.K**2
MERLIVAT_NIEF_1967_ALPHA_I_2H_T1 = 0 * si.K
MERLIVAT_NIEF_1967_ALPHA_I_2H_T0 = -0.0945

""" [Lamb et al. (2017)](https://doi.org/10.1073/pnas.1618374114) """
LAMB_ET_AL_2017_ALPHA_I_2H_T2 = 13525 * si.K**2
LAMB_ET_AL_2017_ALPHA_I_2H_T1 = 0 * si.K
LAMB_ET_AL_2017_ALPHA_I_2H_T0 = -0.0559

""" [Ellehoj et al. (2013)](https://doi.org/10.1002/rcm.6668) """
ELLEHOJ_ET_AL_2013_ALPHA_I_2H_T2 = 48888 * si.K**2
ELLEHOJ_ET_AL_2013_ALPHA_I_2H_T1 = -203.1 * si.K
ELLEHOJ_ET_AL_2013_ALPHA_I_2H_T0 = 0.2133

""" [Majoube (1971)](https://doi.org/10.1051/jcp/1971681423) """
# values taken from Jouzel 1986 (eqs. 1)
MAJOUBE_1971_ALPHA_L_18O_T2 = 1137 * si.K**2
MAJOUBE_1971_ALPHA_L_18O_T1 = -0.4156 * si.K
MAJOUBE_1971_ALPHA_L_18O_T0 = -0.0020667

# values taken from Jouzel 1986 (eqs. 2)
MAJOUBE_1971_ALPHA_L_2H_T2 = 24844 * si.K**2
MAJOUBE_1971_ALPHA_L_2H_T1 = -76.248 * si.K
MAJOUBE_1971_ALPHA_L_2H_T0 = 0.052612

""" [Majoube (1970)](https://doi.org/10.1038/2261242a0) """
MAJOUBE_1970_ALPHA_I_18O_T2 = 0 * si.K**2
MAJOUBE_1970_ALPHA_I_18O_T1 = 11.839 * si.K
MAJOUBE_1970_ALPHA_I_18O_T0 = -0.028224

""" [Van Hook (1968)](https://doi.org/10.1021/j100850a028) """
VAN_HOOK_1968_ALPHA_I_2H_A = 11484.5 * si.K**2
VAN_HOOK_1968_ALPHA_I_2H_B = 35.3315 * si.K
VAN_HOOK_1968_ALPHA_I_2H_C = -0.159290

VAN_HOOK_1968_ALPHA_L_2H_A = 26398.8 * si.K**2
VAN_HOOK_1968_ALPHA_L_2H_B = -89.6065 * si.K
VAN_HOOK_1968_ALPHA_L_2H_C = 0.075802

VAN_HOOK_1968_ALPHA_I_18O_A = 1740.59 * si.K**2
VAN_HOOK_1968_ALPHA_I_18O_B = 2.2965 * si.K
VAN_HOOK_1968_ALPHA_I_18O_C = -0.005793

VAN_HOOK_1968_ALPHA_L_18O_A = 1991.1 * si.K**2
VAN_HOOK_1968_ALPHA_L_18O_B = -4.1887 * si.K
VAN_HOOK_1968_ALPHA_L_18O_C = 0.001197

VAN_HOOK_1968_ALPHA_I_17O_A = 933.651 * si.K**2
VAN_HOOK_1968_ALPHA_I_17O_B = 1.0953 * si.K
VAN_HOOK_1968_ALPHA_I_17O_C = -0.002805

VAN_HOOK_1968_ALPHA_L_17O_A = 1057.8 * si.K**2
VAN_HOOK_1968_ALPHA_L_17O_B = -2.24 * si.K
VAN_HOOK_1968_ALPHA_L_17O_C = 0.000668

VAN_HOOK_1968_ALPHA_I_3H_A = 18464.5 * si.K**2
VAN_HOOK_1968_ALPHA_I_3H_B = 31.0436 * si.K
VAN_HOOK_1968_ALPHA_I_3H_C = -0.20752

VAN_HOOK_1968_ALPHA_L_3H_A = 37813.2 * si.K**2
VAN_HOOK_1968_ALPHA_L_3H_B = -136.751 * si.K
VAN_HOOK_1968_ALPHA_L_3H_C = 0.124096

VAN_HOOK_1968_ALPHA_I_TOT_A = 33453.7 * si.K**2
VAN_HOOK_1968_ALPHA_I_TOT_B = 62.4058 * si.K
VAN_HOOK_1968_ALPHA_I_TOT_C = -0.395542

VAN_HOOK_1968_ALPHA_L_TOT_A = 68702.3 * si.K**2
VAN_HOOK_1968_ALPHA_L_TOT_B = -244.687 * si.K
VAN_HOOK_1968_ALPHA_L_TOT_C = 0.224388

VAN_HOOK_1968_ALPHA_I_DOT_A = 27722.4 * si.K**2
VAN_HOOK_1968_ALPHA_I_DOT_B = 66.5930 * si.K
VAN_HOOK_1968_ALPHA_I_DOT_C = -0.351698

VAN_HOOK_1968_ALPHA_L_DOT_A = 59313.4 * si.K**2
VAN_HOOK_1968_ALPHA_L_DOT_B = -204.941 * si.K
VAN_HOOK_1968_ALPHA_L_DOT_C = 0.182686

VAN_HOOK_1968_ALPHA_I_DOD_A = 21577.6 * si.K**2
VAN_HOOK_1968_ALPHA_I_DOD_B = 69.3358 * si.K
VAN_HOOK_1968_ALPHA_I_DOD_C = -0.305394

VAN_HOOK_1968_ALPHA_L_DOD_A = 49314.9 * si.K**2
VAN_HOOK_1968_ALPHA_L_DOD_B = -164.266 * si.K
VAN_HOOK_1968_ALPHA_L_DOD_C = 0.140049

""" [Horita and Wesolowski (1994)](https://doi.org/10.1016/0016-7037(94)90096-5) """
HORITA_AND_WESOLOWSKI_1994_ALPHA_L_18O_T3 = 1e-3 * 0.35041e9 * si.K**3
HORITA_AND_WESOLOWSKI_1994_ALPHA_L_18O_T2 = 1e-3 * -1.6664e6 * si.K**2
HORITA_AND_WESOLOWSKI_1994_ALPHA_L_18O_T1 = 1e-3 * 6.7123e3 * si.K
HORITA_AND_WESOLOWSKI_1994_ALPHA_L_18O_T0 = 1e-3 * -7.685

HORITA_AND_WESOLOWSKI_1994_ALPHA_L_2H_T3 = 1e-3 * 2.9992e9 * si.K**3
HORITA_AND_WESOLOWSKI_1994_ALPHA_L_2H_T_0 = 1e-3 * -161.04
HORITA_AND_WESOLOWSKI_1994_ALPHA_L_2H_T_1 = 1e-3 * 794.84e-3 / si.K
HORITA_AND_WESOLOWSKI_1994_ALPHA_L_2H_T_2 = 1e-3 * -1620.1e-6 / si.K**2
HORITA_AND_WESOLOWSKI_1994_ALPHA_L_2H_T_3 = 1e-3 * 1158.8e-9 / si.K**3

""" [Barkan and Luz 2005](https://doi.org/10.1002/rcm.2250) """
BARKAN_AND_LUZ_2005_EXPONENT = 0.529

""" eq. 11 in [Barkan and Luz 2007](https://doi.org/10.1002/rcm.3180) """
BARKAN_AND_LUZ_2007_EXCESS_18O_COEFF = 0.528

""" [Craig 1961](https://doi.org/10.1126/science.133.3465.170) """
CRAIG_1961_SLOPE_COEFF = 8
CRAIG_1961_INTERCEPT_COEFF = 10 * PER_MILLE

""" [Bohren 1987](https://doi.org/10.1119/1.15109) """
asymmetry_g = 0.85  # forward scattering from cloud droplets


def compute_derived_values(c: dict):
    c["eps"] = c["Mv"] / c["Md"]
    c["Rd"] = c["R_str"] / c["Md"]
    c["Rv"] = c["R_str"] / c["Mv"]

    c["Rd_over_c_pd"] = c["Rd"] / c["c_pd"]

    c["water_molar_volume"] = c["Mv"] / c["rho_w"]
    c["rho_STP"] = c["p_STP"] / c["Rd"] / c["T_STP"]
    c["H_u"] = c["M"] / c["p_STP"]
