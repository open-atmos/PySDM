# pylint: disable=missing-module-docstring
import numpy as np
import pytest
from chempy import Substance

# from PySDM.initialisation import spectra
from PySDM import Formulae
from PySDM.initialisation.aerosol_composition import DryAerosolMixture
from PySDM.physics import si


@pytest.mark.parametrize(
    "mass_fractions",
    [
        pytest.param({"(NH4)2SO4": 1.0, "insoluble": 0.0}),
        pytest.param({"(NH4)2SO4": 0.5, "insoluble": 0.5}),
        pytest.param({"(NH4)2SO4": 0.0, "insoluble": 1.0}),
    ],
)
def test_volume_weighted_kappa_with_insoluble_compound(mass_fractions):
    # Arrange
    const = Formulae().constants
    water_molar_volume = const.Mv / const.rho_w
    compounds = ("(NH4)2SO4", "insoluble")
    molar_masses = {
        "(NH4)2SO4": Substance.from_formula("(NH4)2SO4").mass * si.gram / si.mole,
        "insoluble": 44 * si.g / si.mole,
    }
    densities = {
        "(NH4)2SO4": 1.77 * si.g / si.cm**3,
        "insoluble": 1.2 * si.g / si.cm**3,
    }
    is_soluble = {"(NH4)2SO4": True, "insoluble": False}
    ionic_dissociation_phi = {"(NH4)2SO4": 3, "insoluble": 0}

    aer = DryAerosolMixture(
        compounds=compounds,
        densities=densities,
        molar_masses=molar_masses,
        is_soluble=is_soluble,
        ionic_dissociation_phi=ionic_dissociation_phi,
    )

    # Act
    kappa_expected = aer.kappa(mass_fractions, water_molar_volume)["Constant"]
    volume_fractions = aer.volume_fractions(mass_fractions)

    # Assert
    compound_kappas = {"(NH4)2SO4": 0.72, "insoluble": 0.0}
    kappa_actual = sum(v * volume_fractions[k] for (k, v) in compound_kappas.items())
    np.testing.assert_approx_equal(
        kappa_expected,
        kappa_actual,
        significant=2,
    )
