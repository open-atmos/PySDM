"""
DryAerosolMixture class defines structure of arbitrary composition aerosol specification.
A DryAerosolMixture must specify the:
    - compounds (Tuple of strings)
    - densities (Dict of [str, float])
    - molar masses (Dict of [str, float]])
    - solubility (Dict of [str, bool])
    - phi, ionic dissociation constant (Dict of [str, int])

DryAerosolMixture class also contains functions for computing
the following properties from `mass_fractions` dictionary:
    - volume fractions
    - soluble volume fraction
    - volume fractions of just soluble or just insoluble species
    - hygroscopicity (kappa) under various assumptions of solubility
    - molar volume of organics, `nu_org`
"""

from typing import Dict, Tuple

from PySDM import formulae
from PySDM.physics import surface_tension
from PySDM.physics.constants_defaults import Mv, rho_w


class DryAerosolMixture:
    def __init__(
        self,
        *,
        compounds: Tuple[str],
        densities: Dict[str, float],
        molar_masses: Dict[str, float],
        is_soluble: Dict[str, bool],
        ionic_dissociation_phi: Dict[str, int]
    ):
        self._aerosol_modes = None
        self.compounds = compounds
        self.densities = densities
        self.molar_masses = molar_masses
        self.is_soluble = is_soluble
        self.ionic_dissociation_phi = ionic_dissociation_phi

    @property
    def aerosol_modes(self):
        return self._aerosol_modes

    @aerosol_modes.setter
    def aerosol_modes(self, value: Tuple[Dict]):
        self._aerosol_modes = value

    # convert mass fractions to volume fractions
    def volume_fractions(self, mass_fractions: dict):
        return {
            k: (mass_fractions[k] / self.densities[k])
            / sum(mass_fractions[i] / self.densities[i] for i in self.compounds)
            for k in self.compounds
        }

    # calculate total volume fraction of soluble species
    def f_soluble_volume(self, mass_fractions: dict):
        volfrac = self.volume_fractions(mass_fractions)
        return sum(self.is_soluble[k] * volfrac[k] for k in self.compounds)

    # calculate volume fractions of just soluble or just insoluble species
    def volfrac_just_soluble(self, volfrac: dict, soluble=True):
        if soluble:
            _masked = {k: (self.is_soluble[k]) * volfrac[k] for k in self.compounds}
        else:
            _masked = {k: (not self.is_soluble[k]) * volfrac[k] for k in self.compounds}

        _denom = sum(list(_masked.values()))
        if _denom == 0.0:
            x = {k: 0.0 for k in self.compounds}
        else:
            x = {k: _masked[k] / _denom for k in self.compounds}
        return x

    # calculate hygroscopicities with different assumptions about solubility
    def kappa(self, mass_fractions: dict):
        volfrac = self.volume_fractions(mass_fractions)
        molar_volumes = {
            i: self.molar_masses[i] / self.densities[i] for i in self.compounds
        }
        volume_fractions_of_just_soluble = self.volfrac_just_soluble(
            volfrac, soluble=True
        )
        all_soluble_ns = sum(
            self.ionic_dissociation_phi[i] * volfrac[i] / molar_volumes[i]
            for i in self.compounds
        )
        part_soluble_ns = self.f_soluble_volume(mass_fractions) * sum(
            self.ionic_dissociation_phi[i]
            * volume_fractions_of_just_soluble[i]
            / molar_volumes[i]
            for i in self.compounds
        )

        result = {}
        for st in formulae._choices(surface_tension).keys():
            if st in (surface_tension.Constant.__name__):
                result[st] = all_soluble_ns * Mv / rho_w
            elif st in (
                surface_tension.CompressedFilmOvadnevaite.__name__,
                surface_tension.CompressedFilmRuehl.__name__,
                surface_tension.SzyszkowskiLangmuir.__name__,
            ):
                result[st] = part_soluble_ns * Mv / rho_w
            else:
                raise AssertionError()
        return result

    # calculate molar volume of just organic species
    def nu_org(self, mass_fractions: dict):
        volfrac = self.volume_fractions(mass_fractions)
        molar_volumes = {
            i: self.molar_masses[i] / self.densities[i] for i in self.compounds
        }
        volume_fractions_of_just_org = self.volfrac_just_soluble(volfrac, soluble=False)
        return sum(
            volume_fractions_of_just_org[i] * molar_volumes[i] for i in self.compounds
        )
