# pylint: disable=missing-module-docstring
import numpy as np
import pytest
from chempy import Substance

# from PySDM.initialisation import spectra
from PySDM.initialisation.aerosol_composition import DryAerosolMixture
from PySDM.physics import si

# spectrum = spectra.Lognormal(norm_factor=100.0 / si.cm**3, m_mode=50.0 * si.nm, s_geom=2.0)


@pytest.mark.parametrize(
    "mass_fractions",
    [
        pytest.param({"(NH4)2SO4": 1.0, "insoluble": 0.0}),
        pytest.param({"(NH4)2SO4": 0.5, "insoluble": 0.5}),
        pytest.param({"(NH4)2SO4": 0.0, "insoluble": 1.0}),
    ],
)
def test_aerosol_init(mass_fractions):
    # Arrange
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
    kappa = aer.kappa(mass_fractions)
    volume_fractions = aer.volume_fractions(mass_fractions)

    # Assert
    kappas = {"(NH4)2SO4": 0.72, "insoluble": 0.0}
    np.testing.assert_approx_equal(
        kappa["Constant"],
        sum(v * volume_fractions[k] for (k, v) in kappas.items()),
        significant=2,
    )
