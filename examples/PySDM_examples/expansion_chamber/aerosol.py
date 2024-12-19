from chempy import Substance
from pystrict import strict

from PySDM.initialisation import spectra
from PySDM.initialisation.aerosol_composition import DryAerosolMixture
from PySDM.physics import si


@strict
class AerosolChamber(DryAerosolMixture):
    def __init__(
        self,
        water_molar_volume: float,
        N: float = 1e5 / si.cm**3,
    ):
        mode1 = {"CaCO3": 1.0}

        super().__init__(
            compounds=("CaCO3",),
            molar_masses={
                "CaCO3": Substance.from_formula("CaCO3").mass * si.g / si.mole
            },
            densities={
                "CaCO3": 2.71 * si.g / si.cm**3,
            },
            is_soluble={
                "CaCO3": True,
            },
            ionic_dissociation_phi={
                "CaCO3": 1,
            },
        )
        self.modes = (
            {
                "kappa": self.kappa(
                    mass_fractions=mode1,
                    water_molar_volume=water_molar_volume,
                ),
                "spectrum": spectra.Lognormal(
                    norm_factor=N,
                    m_mode=158 * si.nm,
                    s_geom=2,
                ),
            },
        )
        # mean diameter 316nm, standard deviation 257nm
        # not sure how to interpret the standard deviation given in the paper
        # because it looks like it's a lognormal distribution in Fig 2
