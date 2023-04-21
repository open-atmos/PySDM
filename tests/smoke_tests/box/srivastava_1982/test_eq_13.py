# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest
from matplotlib import pyplot
from PySDM_examples.Srivastava_1982 import (
    Settings,
    SimProducts,
    coalescence_and_breakup_eq13,
)

from PySDM.physics import si

COMMON_PARAMS = {
    "drop_mass_0": 1 * si.g,
    "dt": 1 * si.s,
    "dv": 1 * si.m**3,
    "total_number": 1e6,
}


@pytest.mark.parametrize(
    "title, settings",
    (
        (
            "merging only",
            Settings(
                srivastava_c=0.5e-6 / si.s,
                srivastava_beta=1e-15 / si.s,
                frag_mass=-1 * si.g,
                n_sds=(8, 128),
                **COMMON_PARAMS
            ),
        ),
        (
            "breakup only",
            Settings(
                srivastava_c=1e-15 / si.s,
                srivastava_beta=1e-9 / si.s,
                frag_mass=0.25 * si.g,
                n_sds=(64, 256),
                **COMMON_PARAMS
            ),
        ),
        (
            "merge + break",
            Settings(
                srivastava_c=0.5e-6 / si.s,
                srivastava_beta=1e-9 / si.s,
                frag_mass=0.25 * si.g,
                n_sds=(2**10, 2**12),
                **COMMON_PARAMS
            ),
        ),
    ),
)
def test_pysdm_coalescence_and_breakup_is_close_to_analytic_coalescence_and_breakup(
    title, settings, plot=False  # TODO #987 (backend_class: CPU, GPU)
):
    n_steps = 256
    results = coalescence_and_breakup_eq13(
        settings, n_steps=n_steps, n_realisations=5, title=title
    )

    if plot:
        pyplot.show()

    # assert
    assert_prod = SimProducts.Computed.mean_drop_volume_total_volume_ratio.name
    np.testing.assert_allclose(
        actual=results.pysdm[settings.n_sds[-1]][assert_prod]["avg"],
        desired=results.analytic[assert_prod],
        rtol=2e-1,
    )
    assert np.mean(results.pysdm[settings.n_sds[-1]][assert_prod]["std"]) < np.mean(
        results.pysdm[settings.n_sds[0]][assert_prod]["std"]
    )

    assert np.mean(
        np.abs(
            results.pysdm[settings.n_sds[-1]][assert_prod]["avg"]
            - results.analytic[assert_prod]
        )
    ) < np.mean(
        np.abs(
            results.pysdm[settings.n_sds[0]][assert_prod]["avg"]
            - results.analytic[assert_prod]
        )
    )
