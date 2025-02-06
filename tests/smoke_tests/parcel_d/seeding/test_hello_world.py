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
        file=Path(seeding.__file__).parent / "hello_world.ipynb",
        plot=PLOT,
    )


class TestHelloWorld:
    @staticmethod
    def test_sd_count(variables):
        minimum = variables["n_sd_initial"]
        maximum = minimum + variables["n_sd_seeding"]
        assert variables["outputs"]["seeding"]["products"]["sd_count"][0] == minimum
        assert (
            minimum
            < variables["outputs"]["seeding"]["products"]["sd_count"][-1]
            < maximum
        )
        np.testing.assert_equal(
            variables["outputs"]["no seeding"]["products"]["sd_count"], minimum
        )

    @staticmethod
    def test_final_rain_water_mixing_ratio_larger_with_seeding(variables):
        assert (
            variables["outputs"]["seeding"]["products"]["rain water mixing ratio"][-1]
            > variables["outputs"]["no seeding"]["products"]["rain water mixing ratio"][
                -1
            ]
        )

    @staticmethod
    def test_rain_water_earlier_with_seeding(variables):
        assert np.count_nonzero(
            variables["outputs"]["seeding"]["products"]["rain water mixing ratio"]
        ) > np.count_nonzero(
            variables["outputs"]["no seeding"]["products"]["rain water mixing ratio"]
        )
