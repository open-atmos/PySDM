import numpy as np
import pytest

from PySDM import Builder
from PySDM.environments import Box


class TestIsotopes:
    @staticmethod
    @pytest.mark.parametrize("isotope", ("2H", "3H", "18O", "17O"))
    @pytest.mark.parametrize("multiplier", (0.1, 1, 10))
    @pytest.mark.parametrize("moles_denom", (0.5, 1, 2))
    def test_delta_attribute(backend_class, isotope, multiplier, moles_denom):
        # arrange
        builder = Builder(n_sd=1, backend=backend_class())
        builder.set_environment(Box(dt=np.nan, dv=np.nan))
        builder.request_attribute(f"delta_{isotope}")

        denom = {"H": "1H", "O": "16O"}[isotope[-1]]
        attributes = {
            f"moles_{denom}": np.asarray([moles_denom]),
            "water mass": np.asarray([np.nan]),
            "multiplicity": np.asarray([-1]),
        }
        attributes[f"moles_{isotope}"] = multiplier * np.asarray(
            [
                getattr(builder.formulae.constants, f"VSMOW_R_{isotope}")
                * attributes[f"moles_{denom}"]
            ]
        )

        particulator = builder.build(attributes=attributes)

        # act
        delta = particulator.attributes[f"delta_{isotope}"].to_ndarray()

        # assert
        np.testing.assert_almost_equal(actual=delta, desired=multiplier - 1)
