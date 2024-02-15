""" regression tests asserting on values from the plots """

from pathlib import Path

import pytest
import numpy as np

from PySDM_examples.utils import notebook_vars
from PySDM_examples import deJong_Azimi

from PySDM.physics import si, in_unit

PLOT = False
RTOL = 1e-2


@pytest.fixture(scope="session", name="variables")
def variables_fixture():
    return notebook_vars(
        file=Path(deJong_Azimi.__file__).parent / "box.ipynb", plot=PLOT
    )


def test_settings_a(variables):
    # Spectra: final spectra
    pysdm_spectra = np.interp(
        variables["r1_plt"],
        variables["res_a"].radius_bins_left_edges,
        variables["res_a"].dv_dlnr[-1] * variables["settings_a"].rho,
    )
    cloudy_a = in_unit(variables["a1_dmdlnr"], si.kg / si.m**3)
    cloudy_b = in_unit(variables["b1_dmdlnr"], si.kg / si.m**3)

    Ea = np.linalg.norm(pysdm_spectra - cloudy_a) / np.linalg.norm(pysdm_spectra)
    Eb = np.linalg.norm(pysdm_spectra - cloudy_b) / np.linalg.norm(pysdm_spectra)
    assert Ea < 1.0
    assert Eb < 1.0

    # Moments: initial M0 and M1
    for i in range(2):
        M_pysdm = in_unit(
            variables["res_a"].moments[:, i]
            * variables["settings_a"].dv
            * variables["settings_a"].rho ** i,
            si.ug**i / si.cm**3,
        )
        M_cloudy_a = deJong_Azimi.cloudy_data_0d.MOM_data["Golovin"]["aMoments"][i]
        M_cloudy_b = (
            deJong_Azimi.cloudy_data_0d.MOM_data["Golovin"]["bMoments"][i, 0]
            + deJong_Azimi.cloudy_data_0d.MOM_data["Golovin"]["bMoments"][i, 1]
        )
        assert np.abs(M_pysdm[0] - M_cloudy_a[0]) / M_pysdm[0] < RTOL
        assert np.abs(M_pysdm[0] - M_cloudy_b[0]) / M_pysdm[0] < RTOL


def test_settings_b(variables):
    # Spectra: final spectra
    pysdm_spectra = np.interp(
        variables["r2_plt"],
        variables["res_b"].radius_bins_left_edges,
        variables["res_b"].dv_dlnr[-1] * variables["settings_b"].rho,
    )
    cloudy_a = in_unit(variables["a2_dmdlnr"], si.kg / si.m**3)
    cloudy_b = in_unit(variables["b2_dmdlnr"], si.kg / si.m**3)

    Ea = np.linalg.norm(pysdm_spectra - cloudy_a) / np.linalg.norm(pysdm_spectra)
    Eb = np.linalg.norm(pysdm_spectra - cloudy_b) / np.linalg.norm(pysdm_spectra)
    assert Ea > 1.0  # large error!
    assert Eb < 1.5  # smaller error :)


def test_settings_c(variables):
    # Moments: initial
    for i in range(3):
        M_pysdm = in_unit(
            variables["res_c"].moments[:, i]
            * variables["settings_c"].dv
            * variables["settings_c"].rho ** i,
            si.ug**i / si.cm**3,
        )
        M_cloudy_a = deJong_Azimi.cloudy_data_0d.MOM_data["Geometric"]["aMoments"][i]
        M_cloudy_b = (
            deJong_Azimi.cloudy_data_0d.MOM_data["Geometric"]["bMoments"][i, 0]
            + deJong_Azimi.cloudy_data_0d.MOM_data["Golovin"]["bMoments"][i, 1]
        )
        assert np.abs(M_pysdm[0] - M_cloudy_a[0]) / M_pysdm[0] < RTOL
        assert np.abs(M_pysdm[0] - M_cloudy_b[0]) / M_pysdm[0] < RTOL
