from chempy import Substance
from pystrict import strict

from PySDM.initialisation import spectra
from PySDM.initialisation.aerosol_composition import DryAerosolMixture
from PySDM.physics import si


@strict
class AerosolBetaCaryophylleneDark(DryAerosolMixture):
    def __init__(self, water_molar_volume: float, Forg: float = 0.8, N: float = 400):
        mode = {
            "(NH4)2SO4": (1 - Forg),
            "bcary_dark": Forg,
        }

        super().__init__(
            compounds=("(NH4)2SO4", "bcary_dark"),
            molar_masses={
                "(NH4)2SO4": Substance.from_formula("(NH4)2SO4").mass
                * si.gram
                / si.mole,
                "bcary_dark": 299 * si.gram / si.mole,
            },
            densities={
                "(NH4)2SO4": 1.77 * si.g / si.cm**3,
                "bcary_dark": 1.20 * si.g / si.cm**3,
            },
            is_soluble={
                "(NH4)2SO4": False,
                "bcary_dark": True,
            },
            ionic_dissociation_phi={
                "(NH4)2SO4": 3,
                "bcary_dark": 1,
            },
        )
        self.modes = (
            {
                "f_org": 1 - self.f_soluble_volume(mode),
                "kappa": self.kappa(
                    mass_fractions=mode, water_molar_volume=water_molar_volume
                ),
                "nu_org": self.nu_org(mode),
                "spectrum": spectra.Lognormal(
                    norm_factor=N / si.cm**3, m_mode=50.0 * si.nm, s_geom=1.75
                ),
            },
        )

    color = "red"


class AerosolBetaCaryophylleneLight(DryAerosolMixture):
    def __init__(self, water_molar_volume: float, Forg: float = 0.8, N: float = 400):
        mode = {
            "(NH4)2SO4": (1 - Forg),
            "bcary_light": Forg,
        }

        super().__init__(
            compounds=("(NH4)2SO4", "bcary_light"),
            molar_masses={
                "(NH4)2SO4": Substance.from_formula("(NH4)2SO4").mass
                * si.gram
                / si.mole,
                "bcary_light": 360 * si.gram / si.mole,
            },
            densities={
                "(NH4)2SO4": 1.77 * si.g / si.cm**3,
                "bcary_light": 1.50 * si.g / si.cm**3,
            },
            is_soluble={
                "(NH4)2SO4": False,
                "bcary_light": True,
            },
            ionic_dissociation_phi={
                "(NH4)2SO4": 3,
                "bcary_light": 1,
            },
        )
        self.modes = (
            {
                "f_org": 1 - self.f_soluble_volume(mode),
                "kappa": self.kappa(
                    mass_fractions=mode, water_molar_volume=water_molar_volume
                ),
                "nu_org": self.nu_org(mode),
                "spectrum": spectra.Lognormal(
                    norm_factor=N / si.cm**3, m_mode=50.0 * si.nm, s_geom=1.75
                ),
            },
        )

    color = "orange"


@strict
class AerosolAlphaPineneDark(DryAerosolMixture):
    def __init__(self, water_molar_volume: float, Forg: float = 0.8, N: float = 400):
        mode = {
            "(NH4)2SO4": (1 - Forg),
            "apinene_dark": Forg,
        }

        super().__init__(
            compounds=("(NH4)2SO4", "apinene_dark"),
            molar_masses={
                "(NH4)2SO4": Substance.from_formula("(NH4)2SO4").mass
                * si.gram
                / si.mole,
                "apinene_dark": 209 * si.gram / si.mole,
            },
            densities={
                "(NH4)2SO4": 1.77 * si.g / si.cm**3,
                "apinene_dark": 1.27 * si.g / si.cm**3,
            },
            is_soluble={
                "(NH4)2SO4": False,
                "apinene_dark": True,
            },
            ionic_dissociation_phi={
                "(NH4)2SO4": 3,
                "apinene_dark": 1,
            },
        )
        self.modes = (
            {
                "f_org": 1 - self.f_soluble_volume(mode),
                "kappa": self.kappa(
                    mass_fractions=mode, water_molar_volume=water_molar_volume
                ),
                "nu_org": self.nu_org(mode),
                "spectrum": spectra.Lognormal(
                    norm_factor=N / si.cm**3, m_mode=50.0 * si.nm, s_geom=1.75
                ),
            },
        )

    color = "green"


@strict
class AerosolAlphaPineneLight(DryAerosolMixture):
    def __init__(self, water_molar_volume: float, Forg: float = 0.8, N: float = 400):
        mode = {
            "(NH4)2SO4": (1 - Forg),
            "apinene_light": Forg,
        }

        super().__init__(
            compounds=("(NH4)2SO4", "apinene_light"),
            molar_masses={
                "(NH4)2SO4": Substance.from_formula("(NH4)2SO4").mass
                * si.gram
                / si.mole,
                "apinene_light": 265 * si.gram / si.mole,
            },
            densities={
                "(NH4)2SO4": 1.77 * si.g / si.cm**3,
                "apinene_light": 1.51 * si.g / si.cm**3,
            },
            is_soluble={
                "(NH4)2SO4": False,
                "apinene_light": True,
            },
            ionic_dissociation_phi={
                "(NH4)2SO4": 3,
                "apinene_light": 1,
            },
        )
        self.modes = (
            {
                "f_org": 1 - self.f_soluble_volume(mode),
                "kappa": self.kappa(
                    mass_fractions=mode, water_molar_volume=water_molar_volume
                ),
                "nu_org": self.nu_org(mode),
                "spectrum": spectra.Lognormal(
                    norm_factor=N / si.cm**3, m_mode=50.0 * si.nm, s_geom=1.75
                ),
            },
        )

    color = "lightgreen"
