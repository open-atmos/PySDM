import numpy as np
import pytest

from PySDM.dynamics import DropLocalThermodynamics, VapourDepositionOnIce
from PySDM import Builder, Formulae
from PySDM.physics import si

from tests.unit_tests.dynamics.test_vapour_deposition_on_ice import MoistBox

# TODO: hello-world scenario: constant timescale, no diffusional growth, pure relaxation, assert that local temp relaxes to ambient according to the tau value


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
def test_drop_local_thermodynamics_e_folding(tau, fraction_at_t0, backend_class):
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
    builder.add_dynamic(DropLocalThermodynamics())
    particulator = builder.build(
        products=(),
        attributes={
            "multiplicity": np.asarray([1, 2, 3]),
            "signed water mass": np.asarray([1, 2, 3]) * si.ng,
            "drop-local water vapour mixing ratio": np.full(n_sd, fraction_at_t0 * r0),
        },
    )
    particulator.environment["water_vapour_mixing_ratio"] = r0

    # act
    particulator.run(steps=1)

    # assert
    r1 = particulator.attributes["drop-local water vapour mixing ratio"].to_ndarray()
    if fraction_at_t0 == 1:
        assert (r1 == r0).all()
    else:
        assert (r1 != r0).all()
    assert ((r1 - r0) / dt == -r0 * (fraction_at_t0 - 1) / tau).all()
