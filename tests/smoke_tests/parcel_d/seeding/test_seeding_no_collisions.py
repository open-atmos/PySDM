"""tests ensuring values on plots match those in the paper"""

from pathlib import Path

import numpy as np
import pytest

from open_atmos_jupyter_utils import notebook_vars
from PySDM_examples import seeding

PLOT = False


@pytest.fixture(scope="session", name="variables")
def variables_fixture():
    return notebook_vars(
        file=Path(seeding.__file__).parent / "seeding_no_collisions.ipynb",
        plot=PLOT,
    )


class TestSeedingNoCollisions:
    @staticmethod
    # seeding has smaller cloud drops than no seeding
    def test_reff(variables):
        np.testing.assert_array_less(
            variables["outputs"]["seeding"]["products"]["r_eff"],
            variables["outputs"]["no seeding"]["products"]["r_eff"] + 1e-6,
        )

    @staticmethod
    # seeding has more cloud drops than no seeding
    def test_n_drop(variables):
        np.testing.assert_array_less(
            variables["outputs"]["no seeding"]["products"]["n_drop"],
            variables["outputs"]["seeding"]["products"]["n_drop"] + 1e-6,
        )
