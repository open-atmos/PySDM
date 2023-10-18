import numpy as np
import pytest

from PySDM import Builder
from PySDM.attributes.isotopes import ISOTOPES
from PySDM.environments import Box


def dummy_attrs(len):
    return {
        "water mass": np.asarray([np.nan] * len),
        "multiplicity": np.asarray([-1] * len, dtype=int),
    }


class TestIsotopes:
    @staticmethod
    @pytest.mark.parametrize("isotope", ISOTOPES)
    def test_moles_attributes(backend_class, isotope):
        # arrange
        values = [1, 2, 3]
        builder = Builder(n_sd=len(values), backend=backend_class())
        builder.set_environment(Box(dt=np.nan, dv=np.nan))
        particulator = builder.build(
            attributes={
                f"moles_{isotope}": np.asarray(values),
                **dummy_attrs(len(values)),
            }
        )

        # act
        attr_values = particulator.attributes[f"moles_{isotope}"].to_ndarray()

        # assert
        np.testing.assert_array_equal(attr_values, values)

    @staticmethod
    @pytest.mark.parametrize(
        "isotope", [iso for iso in ISOTOPES if iso not in ("1H", "16O")]
    )
    @pytest.mark.parametrize("multiplier", (0.01, 1, 100))
    @pytest.mark.parametrize("moles_denom", (0.5, 1, 2))
    def test_delta_attribute(backend_class, isotope, multiplier, moles_denom):
        # arrange
        builder = Builder(n_sd=1, backend=backend_class(double_precision=True))
        builder.set_environment(Box(dt=np.nan, dv=np.nan))
        builder.request_attribute(f"delta_{isotope}")

        denom = {"H": "1H", "O": "16O"}[isotope[-1]]
        attributes = {
            f"moles_{denom}": np.asarray([moles_denom]),
            **dummy_attrs(1),
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
