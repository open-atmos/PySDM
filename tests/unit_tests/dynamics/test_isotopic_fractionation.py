"""
unit test for the IsotopicFractionation dynamic
"""

from contextlib import nullcontext

import numpy as np
import pytest

from PySDM import Builder
from PySDM.dynamics import Condensation, IsotopicFractionation
from PySDM.dynamics.isotopic_fractionation import HEAVY_ISOTOPES
from PySDM.environments import Box
from PySDM.physics import si


@pytest.mark.parametrize(
    "dynamics, context",
    (
        pytest.param(
            (Condensation(), IsotopicFractionation(isotopes=("1H",))), nullcontext()
        ),
        pytest.param(
            (IsotopicFractionation(isotopes=("1H",)),),
            pytest.raises(AssertionError, match="dynamics"),
        ),
        pytest.param(
            (IsotopicFractionation(isotopes=("1H",)), Condensation()),
            pytest.raises(AssertionError, match="dynamics"),
        ),
    ),
)
def test_ensure_condensation_executed_before(backend_class, dynamics, context):
    # arrange
    builder = Builder(
        n_sd=1, backend=backend_class(), environment=Box(dv=np.nan, dt=1 * si.s)
    )
    for dynamic in dynamics:
        builder.add_dynamic(dynamic)

    # act
    with context:
        builder.build(
            attributes={
                attr: np.asarray([np.nan if attr != "multiplicity" else 0])
                for attr in (
                    "multiplicity",
                    "water mass",
                    "dry volume",
                    "kappa times dry volume",
                    *[f"moles_{isotope}" for isotope in HEAVY_ISOTOPES],
                )
            }
        )
