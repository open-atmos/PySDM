from chempy import Substance
from pystrict import strict

from PySDM.initialisation import spectra
from PySDM.initialisation.aerosol_composition import DryAerosolMixture
from PySDM.physics import si


@strict
class AerosolMarine(DryAerosolMixture):
    def __init__(
        self, water_molar_volume: float, Forg: float = 0.2, Acc_N2: float = 134
    ):
        Aitken = {
            "palmitic": Forg,
            "(NH4)2SO4": (1 - Forg),
            "NaCl": 0,
        }
        Accumulation = {
            "palmitic": Forg,
            "(NH4)2SO4": 0,
            "NaCl": (1 - Forg),
        }

        super().__init__(
            ionic_dissociation_phi={
                "palmitic": 1,
                "(NH4)2SO4": 3,
                "NaCl": 2,
            },
            is_soluble={
                "palmitic": False,
                "(NH4)2SO4": True,
                "NaCl": True,
            },
            densities={
                "palmitic": 0.852 * si.g / si.cm**3,
                "(NH4)2SO4": 1.77 * si.g / si.cm**3,
                "NaCl": 2.16 * si.g / si.cm**3,
            },
            compounds=("palmitic", "(NH4)2SO4", "NaCl"),
            molar_masses={
                "palmitic": 256.4 * si.g / si.mole,
                "(NH4)2SO4": Substance.from_formula("(NH4)2SO4").mass
                * si.gram
                / si.mole,
                "NaCl": Substance.from_formula("NaCl").mass * si.gram / si.mole,
            },
        )
        self.modes = (
            {
                "f_org": 1 - self.f_soluble_volume(Aitken),
                "kappa": self.kappa(Aitken, water_molar_volume=water_molar_volume),
                "nu_org": self.nu_org(Aitken),
                "spectrum": spectra.Lognormal(
                    norm_factor=226 / si.cm**3, m_mode=19.6 * si.nm, s_geom=1.71
                ),
            },
            {
                "f_org": 1 - self.f_soluble_volume(Accumulation),
                "kappa": self.kappa(
                    Accumulation, water_molar_volume=water_molar_volume
                ),
                "nu_org": self.nu_org(Accumulation),
                "spectrum": spectra.Lognormal(
                    norm_factor=Acc_N2 / si.cm**3, m_mode=69.5 * si.nm, s_geom=1.7
                ),
            },
        )

    color = "dodgerblue"


@strict
class AerosolBoreal(DryAerosolMixture):
    def __init__(
        self, water_molar_volume: float, Forg: float = 0.668, Acc_N2: float = 540
    ):
        # note: SOA1 or SOA2 unclear from the paper
        Aitken = {
            "SOA1": Forg,
            "SOA2": 0,
            "(NH4)2SO4": (1 - Forg) / 2,
            "NH4NO3": (1 - Forg) / 2,
        }
        Accumulation = {
            "SOA1": 0,
            "SOA2": Forg,
            "(NH4)2SO4": (1 - Forg) / 2,
            "NH4NO3": (1 - Forg) / 2,
        }

        super().__init__(
            ionic_dissociation_phi={
                "SOA1": 1,
                "SOA2": 1,
                "(NH4)2SO4": 3,
                "NH4NO3": 2,
            },
            molar_masses={
                "(NH4)2SO4": Substance.from_formula("(NH4)2SO4").mass
                * si.gram
                / si.mole,
                "NH4NO3": Substance.from_formula("NH4NO3").mass * si.gram / si.mole,
                "SOA1": 190 * si.g / si.mole,
                "SOA2": 368.4 * si.g / si.mole,
            },
            densities={
                "SOA1": 1.24 * si.g / si.cm**3,
                "SOA2": 1.2 * si.g / si.cm**3,
                "(NH4)2SO4": 1.77 * si.g / si.cm**3,
                "NH4NO3": 1.72 * si.g / si.cm**3,
            },
            compounds=("SOA1", "SOA2", "(NH4)2SO4", "NH4NO3"),
            is_soluble={
                "SOA1": False,
                "SOA2": False,
                "(NH4)2SO4": True,
                "NH4NO3": True,
            },
        )
        self.modes = (
            {
                "f_org": 1 - self.f_soluble_volume(Aitken),
                "kappa": self.kappa(Aitken, water_molar_volume=water_molar_volume),
                "nu_org": self.nu_org(Aitken),
                "spectrum": spectra.Lognormal(
                    norm_factor=1110 / si.cm**3, m_mode=22.7 * si.nm, s_geom=1.75
                ),
            },
            {
                "f_org": 1 - self.f_soluble_volume(Accumulation),
                "kappa": self.kappa(
                    Accumulation, water_molar_volume=water_molar_volume
                ),
                "nu_org": self.nu_org(Accumulation),
                "spectrum": spectra.Lognormal(
                    norm_factor=Acc_N2 / si.cm**3,
                    m_mode=82.2 * si.nm,
                    s_geom=1.62,
                ),
            },
        )

    color = "yellowgreen"


@strict
class AerosolNascent(DryAerosolMixture):
    def __init__(
        self, water_molar_volume: float, Acc_Forg: float = 0.3, Acc_N2: float = 30
    ):
        Ultrafine = {
            "SOA1": 0.52,
            "SOA2": 0,
            "(NH4)2SO4": 0.48,
        }
        Accumulation = {
            "SOA1": 0,
            "SOA2": Acc_Forg,
            "(NH4)2SO4": (1 - Acc_Forg),
        }
        super().__init__(
            ionic_dissociation_phi={
                "SOA1": 1,
                "SOA2": 1,
                "(NH4)2SO4": 3,
            },
            molar_masses={
                "SOA1": 190 * si.g / si.mole,
                "SOA2": 368.4 * si.g / si.mole,
                "(NH4)2SO4": Substance.from_formula("(NH4)2SO4").mass
                * si.gram
                / si.mole,
            },
            densities={
                "SOA1": 1.24 * si.g / si.cm**3,
                "SOA2": 1.2 * si.g / si.cm**3,
                "(NH4)2SO4": 1.77 * si.g / si.cm**3,
            },
            compounds=("SOA1", "SOA2", "(NH4)2SO4"),
            is_soluble={
                "SOA1": False,
                "SOA2": False,
                "(NH4)2SO4": True,
            },
        )
        self.modes = (
            {
                "f_org": 1 - self.f_soluble_volume(Ultrafine),
                "kappa": self.kappa(Ultrafine, water_molar_volume=water_molar_volume),
                "nu_org": self.nu_org(Ultrafine),
                "spectrum": spectra.Lognormal(
                    norm_factor=2000 / si.cm**3, m_mode=11.5 * si.nm, s_geom=1.71
                ),
            },
            {
                "f_org": 1 - self.f_soluble_volume(Accumulation),
                "kappa": self.kappa(
                    Accumulation, water_molar_volume=water_molar_volume
                ),
                "nu_org": self.nu_org(Accumulation),
                "spectrum": spectra.Lognormal(
                    norm_factor=Acc_N2 / si.cm**3, m_mode=100 * si.nm, s_geom=1.70
                ),
            },
        )

    color = "orangered"
