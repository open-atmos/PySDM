"""
    test for supplementary figure 2 in Lowe et al 2019 paper.
    checks that values from panels d)-f) are in a reasonable range
    and decrease/increase monotonically with updraft velocity
"""

import os
from pathlib import Path

import numpy as np
import pytest
from PySDM_examples import Lowe_et_al_2019
from PySDM_examples.utils import notebook_vars

from PySDM.physics import si

PLOT = False


@pytest.fixture(scope="session", name="variables")
def variables_fixture():
    return notebook_vars(
        file=Path(Lowe_et_al_2019.__file__).parent / "fig_s2.ipynb", plot=PLOT
    )


CI = "CI" in os.environ
nRes = 10
updrafts = np.linspace(0.2, 2.4, 2 if CI else nRes)
models = ("Constant", "CompressedFilmOvadnevaite")
aerosol_names = ("AerosolMarine", "AerosolBoreal", "AerosolNascent")


def keygen(updraft, model, aerosol_class_name):
    return f"w{updraft:.2f}_{aerosol_class_name}_{model}"


@pytest.fixture(
    params=[
        keygen(updraft, model, aerosol_class_name)
        for updraft in updrafts
        for model in models
        for aerosol_class_name in aerosol_names
    ],
    scope="session",
    name="key",
)
def keys_fixture(request):
    return request.param


@pytest.fixture(params=models, scope="session", name="model")
def models_fixture(request):
    return request.param


@pytest.fixture(params=aerosol_names, scope="session", name="aerosol_class_name")
def aerosols_fixture(request):
    return request.param


class TestFigS2:
    @staticmethod
    @pytest.mark.parametrize(
        "var, value_range",
        (
            ("lwp", (28 * si.g / si.m**2, 36 * si.g / si.m**2)),  # TODO #1247: 28 to 33
            ("tau", (2, 13)),  # TODO #1247: 2 to 11
            ("albedo", (0.15, 0.5)),  # TODO #1247: 0.15 to 0.45
        ),
    )
    # TODO #1246: range mismatch possibly related to supersaturation profile discrepancies
    def test_ranges(var, value_range, variables, key):
        assert value_range[0] < variables["output"][key][var] < value_range[1]

    @staticmethod
    @pytest.mark.parametrize("var, sgn", (("lwp", -1), ("tau", 1), ("albedo", 1)))
    def test_monotonicity(var, sgn, variables, model, aerosol_class_name):
        tmp = [
            variables["output"][keygen(updraft, model, aerosol_class_name)][var]
            for updraft in updrafts
        ]
        assert (np.diff(tmp) * sgn > -np.mean(tmp) * 1e-2).all()
