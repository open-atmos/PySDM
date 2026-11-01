from chempy import Substance
from pystrict import strict

from PySDM.initialisation import spectra
from PySDM.initialisation.aerosol_composition import DryAerosolMixture
from PySDM.physics import si


@strict
class AerosolMarine(DryAerosolMixture):
    # cd MAV
    # MODAL_PARS.CONC         = [223  137];   %[number/cm3]
    # MODAL_PARS.GSD          = [1.68 1.68];
    # MODAL_PARS.GEOMEAN_DIAM = [0.0390  0.139];     %[um]
    # DENSITY(4:5)            = 0.852;  %Set organic density. Palmitic acid 256.4 [gmol-1]
    # MASS_FRAC               = [0 0.8 0 0.2 0 0 0 0.0;...
    #                            0 0.0 0 0.2 0 0 0 0.8];
    # NAT                     = 2;
    # NMODE                   = [1 1];
    # DENSITY                 = [1.841, 1.78, 1.77, 0.852, 1.5, 2., 2.65, 2.165]; %[gcm-3]

    def __init__(
        self, water_molar_volume: float, Forg: float = 0.2, Acc_N2: float = 137
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
                "(NH4)2SO4": 1.78 * si.g / si.cm**3,
                "NaCl": 2.165 * si.g / si.cm**3,
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
                "kappa": self.kappa(
                    mass_fractions=Aitken, water_molar_volume=water_molar_volume
                ),
                "nu_org": self.nu_org(Aitken),
                "spectrum": spectra.Lognormal(
                    norm_factor=223 / si.cm**3, m_mode=19.6 * si.nm, s_geom=1.68
                ),
            },
            {
                "f_org": 1 - self.f_soluble_volume(Accumulation),
                "kappa": self.kappa(
                    mass_fractions=Accumulation, water_molar_volume=water_molar_volume
                ),
                "nu_org": self.nu_org(Accumulation),
                "spectrum": spectra.Lognormal(
                    norm_factor=Acc_N2 / si.cm**3, m_mode=69.5 * si.nm, s_geom=1.68
                ),
            },
        )

    color = "dodgerblue"


@strict
class AerosolBoreal(DryAerosolMixture):
    #     cd HYY
    #     MODAL_PARS.CONC         = [1110   540];
    #     MODAL_PARS.GSD          = [1.75   1.62];
    #     MODAL_PARS.GEOMEAN_DIAM = [0.0453 0.1644];
    #     DENSITY(4:5)            = [1.2 1.4];                  %Set organic density
    #     DENSITY(1)              = 1.72;
    #     INORG_MASS_RATIO        = 0.1515/0.1559;  %ammonium sulfate:ammonium nitrate  mass
    #     FORG                    = 0.60;
    #     MASS_FRAC               = [(1-FORG)/(1+INORG_MASS_RATIO) 0 ...
    #                               INORG_MASS_RATIO*(1-FORG)/(1+INORG_MASS_RATIO) FORG...
    #                               0 0 0 0;...
    #                               (1-FORG)/(1+INORG_MASS_RATIO) 0 ...
    #                               INORG_MASS_RATIO*(1-FORG)/(1+INORG_MASS_RATIO) 0 ...
    #                               FORG 0 0 0];
    #     NAT                     = 2;
    #     NMODE                   = [1 1];

    # DENSITY                     = [1.841, 1.78, 1.77, 1.5, 1.5, 2., 2.65, 2.165]; %[gcm-3]
    # DENSITY                     = [1.72, 1.78, 1.77, 1.2, 1.4, 2., 2.65, 2.165]; %[gcm-3]

    def __init__(
        self, water_molar_volume: float, Forg: float = 0.668, Acc_N2: float = 540
    ):
        # TODO #1247: SOA1 or SOA2 unclear from the paper
        # TODO #1247: CAN'T FIND WHERE NH4NO3 PROPERTIES ARE DEFINED IN ICPM
        INORG_MASS_RATIO = 0.1515 / 0.1559
        Aitken = {
            "SOA1": Forg,
            "SOA2": 0,
            "(NH4)2SO4": INORG_MASS_RATIO * (1 - Forg) / (1 + INORG_MASS_RATIO),
            "NH4NO3": (1 - Forg) / (1 + INORG_MASS_RATIO),
        }
        Accumulation = {
            "SOA1": 0,
            "SOA2": Forg,
            "(NH4)2SO4": INORG_MASS_RATIO * (1 - Forg) / (1 + INORG_MASS_RATIO),
            "NH4NO3": (1 - Forg) / (1 + INORG_MASS_RATIO),
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
                "SOA1": 190 * si.g / si.mole,  # TODO #1247: 190 OR 200?
                "SOA2": 368.4 * si.g / si.mole,  # TODO #1247: 368.4 OR 200?
            },
            densities={
                "SOA1": 1.2 * si.g / si.cm**3,
                "SOA2": 1.4 * si.g / si.cm**3,
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
                "kappa": self.kappa(
                    mass_fractions=Aitken, water_molar_volume=water_molar_volume
                ),
                "nu_org": self.nu_org(Aitken),
                "spectrum": spectra.Lognormal(
                    norm_factor=1110 / si.cm**3, m_mode=22.65 * si.nm, s_geom=1.75
                ),
            },
            {
                "f_org": 1 - self.f_soluble_volume(Accumulation),
                "kappa": self.kappa(
                    mass_fractions=Accumulation, water_molar_volume=water_molar_volume
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
    # cd NUM
    # MODAL_PARS.CONC         = [2000  30];
    # MODAL_PARS.GSD          = [1.71  1.703];
    # MODAL_PARS.GEOMEAN_DIAM = [0.023 0.200];
    # DENSITY(4:5)            = [1.2 1.24];                  %Set organic density
    # MASS_FRAC               = [0 0 0.48 0.52 0 0 0 0;...
    #                            0 0 0.70 0.0 0.3 0 0 0];
    # NAT                     = 2;
    # NMODE                   = [1 1];

    # DENSITY                 = [1.841, 1.78, 1.77, 1.2, 1.24, 2., 2.65, 2.165]; %[gcm-3]

    def __init__(
        self, water_molar_volume: float, Acc_Forg: float = 0.3, Acc_N2: float = 30
    ):
        # TODO #1247: CAN'T FIND WHEN PHI IS MULTIPLIED FOR KÖHLER B IN ICPM CODE
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
                "SOA1": 190 * si.g / si.mole,  # TODO #1247: 190 OR 200?
                "SOA2": 368.4 * si.g / si.mole,  # TODO #1247: 368.4 OR 200?
                "(NH4)2SO4": Substance.from_formula("(NH4)2SO4").mass
                * si.gram
                / si.mole,
            },
            densities={
                "SOA1": 1.2 * si.g / si.cm**3,
                "SOA2": 1.24 * si.g / si.cm**3,
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
                "kappa": self.kappa(
                    mass_fractions=Ultrafine, water_molar_volume=water_molar_volume
                ),
                "nu_org": self.nu_org(Ultrafine),
                "spectrum": spectra.Lognormal(
                    norm_factor=2000 / si.cm**3, m_mode=11.5 * si.nm, s_geom=1.71
                ),
            },
            {
                "f_org": 1 - self.f_soluble_volume(Accumulation),
                "kappa": self.kappa(
                    mass_fractions=Accumulation, water_molar_volume=water_molar_volume
                ),
                "nu_org": self.nu_org(Accumulation),
                "spectrum": spectra.Lognormal(
                    norm_factor=Acc_N2 / si.cm**3, m_mode=100 * si.nm, s_geom=1.703
                ),
            },
        )

    color = "orangered"
