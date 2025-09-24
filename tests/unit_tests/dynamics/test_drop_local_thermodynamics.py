import numpy as np
import pytest

from PySDM.dynamics import (
    DropLocalThermodynamics,
    VapourDepositionOnIce,
    AmbientThermodynamics,
)
from PySDM import Builder, Formulae
from PySDM.physics import si

from tests.unit_tests.dynamics.test_vapour_deposition_on_ice import MoistBox


@pytest.mark.parametrize(
    "tau, fraction_at_t0",
    (
        (123 * si.s, 1),
        (666 * si.s, 0.9),
        (44 * si.s, 0.9),
        (666 * si.s, 1.1),
        (44 * si.s, 1.1),
    ),
)
def test_drop_local_thermodynamics_for_constant_tau(tau, fraction_at_t0, backend_class):
    if backend_class.__name__ == "ThrustRTC":
        pytest.skip()  # TODO

    # arrange
    n_sd = 3
    r0 = 12.345 * si.g / si.kg
    dt = 0.6789 * si.s

    formulae = Formulae(
        turbulent_relaxation_timescale="Constant",
        constants={"TURBULENT_RELAXATION_TIMESCALE_FOR_TESTS": tau},
    )
    builder = Builder(
        n_sd=n_sd,
        backend=backend_class(formulae),
        environment=MoistBox(dt=dt, dv=np.nan),
    )
    builder.add_dynamic(AmbientThermodynamics())
    builder.add_dynamic(DropLocalThermodynamics())
    particulator = builder.build(
        products=(),
        attributes={
            "multiplicity": np.asarray([1, 2, 3]),
            "signed water mass": np.asarray([1, 2, 3]) * si.ng,
        },
    )
    particulator.environment["water_vapour_mixing_ratio"] = np.nan
    particulator.environment["thd"] = np.nan

    # act
    particulator.run(steps=0)
    particulator.environment["water_vapour_mixing_ratio"] = r0
    particulator.attributes["dropwise water vapour mixing ratio"].data[:] = (
        fraction_at_t0 * r0
    )
    particulator.run(steps=1)

    # assert
    r1 = particulator.attributes["dropwise water vapour mixing ratio"].to_ndarray()
    if fraction_at_t0 == 1:
        assert (r1 == r0).all()
    else:
        assert (r1 != r0).all()
    np.testing.assert_allclose(
        actual=(r1 - fraction_at_t0 * r0) / dt,
        desired=-r0 * (fraction_at_t0 - 1) / tau,
    )
