"""
Planetary properties used in calculations.

Values are primarily taken from Table 1 of Loftus & Wordsworth (2021),
unless otherwise noted.
Each variable represents a physical property or atmospheric
composition relevant for cloud microphysics modeling.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any

from PySDM.physics.constants import si


@dataclass
class Planet:
    g_std: float
    T_STP: float
    p_STP: float
    RH_zref: float
    dry_molar_conc_H2: float
    dry_molar_conc_He: float
    dry_molar_conc_N2: float
    dry_molar_conc_O2: float
    dry_molar_conc_CO2: float
    H_LCL: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return vars(self)


@dataclass
class EarthLike(Planet):
    g_std: float = 9.82 * si.metre / si.second**2
    T_STP: float = 300 * si.kelvin
    p_STP: float = 1.01325 * 1e5 * si.Pa
    RH_zref: float = 0.75
    dry_molar_conc_H2: float = 0
    dry_molar_conc_He: float = 0
    dry_molar_conc_N2: float = 1
    dry_molar_conc_O2: float = 0
    dry_molar_conc_CO2: float = 0
    H_LCL: float = 8.97 * si.kilometre


@dataclass
class Earth(Planet):
    g_std: float = 9.82 * si.metre / si.second**2
    T_STP: float = 290 * si.kelvin
    p_STP: float = 1.01325 * 1e5 * si.Pa
    RH_zref: float = 0.75
    dry_molar_conc_H2: float = 0
    dry_molar_conc_He: float = 0
    dry_molar_conc_N2: float = 0.8
    dry_molar_conc_O2: float = 0.2
    dry_molar_conc_CO2: float = 0
    H_LCL: float = 8.41 * si.kilometre


@dataclass
class EarlyMars(Planet):
    g_std: float = 3.71 * si.metre / si.second**2
    T_STP: float = 290 * si.kelvin
    p_STP: float = 2 * 1e5 * si.Pa
    RH_zref: float = 0.75
    dry_molar_conc_H2: float = 0
    dry_molar_conc_He: float = 0
    dry_molar_conc_N2: float = 0
    dry_molar_conc_O2: float = 0
    dry_molar_conc_CO2: float = 1
    H_LCL: float = 14.5 * si.kilometre


@dataclass
class Jupiter(Planet):
    g_std: float = 24.84 * si.metre / si.second**2
    T_STP: float = 274 * si.kelvin
    p_STP: float = 4.85 * 1e5 * si.Pa
    RH_zref: float = 1
    dry_molar_conc_H2: float = 0.864
    dry_molar_conc_He: float = 0.136
    dry_molar_conc_N2: float = 0
    dry_molar_conc_O2: float = 0
    dry_molar_conc_CO2: float = 0
    H_LCL: float = 39.8 * si.kilometre


@dataclass
class Saturn(Planet):
    g_std: float = 10.47 * si.metre / si.second**2
    T_STP: float = 284 * si.kelvin
    p_STP: float = 10.4 * 1e5 * si.Pa
    RH_zref: float = 1
    dry_molar_conc_H2: float = 0.88
    dry_molar_conc_He: float = 0.12
    dry_molar_conc_N2: float = 0
    dry_molar_conc_O2: float = 0
    dry_molar_conc_CO2: float = 0
    H_LCL: float = 99.2 * si.kilometre


@dataclass
class K2_18B(Planet):
    g_std: float = 12.44 * si.metre / si.second**2
    T_STP: float = 275 * si.kelvin
    p_STP: float = 0.1 * 1e5 * si.Pa
    RH_zref: float = 1
    dry_molar_conc_H2: float = 0.9
    dry_molar_conc_He: float = 0.1
    dry_molar_conc_N2: float = 0
    dry_molar_conc_O2: float = 0
    dry_molar_conc_CO2: float = 0
    H_LCL: float = 56.6 * si.kilometre


@dataclass
class CompositeTest(Planet):
    g_std: float = 9.82 * si.metre / si.second**2
    T_STP: float = 275 * si.kelvin
    p_STP: float = 0.75 * 1e5 * si.Pa
    RH_zref: float = 1
    dry_molar_conc_H2: float = 0.1
    dry_molar_conc_He: float = 0.1
    dry_molar_conc_N2: float = 0.1
    dry_molar_conc_O2: float = 0.1
    dry_molar_conc_CO2: float = 0.1
    H_LCL: Optional[float] = None
