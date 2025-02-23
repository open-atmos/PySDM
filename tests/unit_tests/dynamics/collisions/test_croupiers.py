# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest

from PySDM.backends import ThrustRTC
from PySDM.initialisation.sampling.spectral_sampling import Linear
from PySDM.initialisation.spectra.lognormal import Lognormal

from ...dummy_particulator import DummyParticulator


@pytest.mark.parametrize("croupier", ["local", "global"])
def test_final_state(croupier, backend_class):
    if backend_class is ThrustRTC:
        pytest.skip("TODO #330")

    # Arrange
    n_part = 100000
    v_mean = 2e-6
    d = 1.2
    n_sd = 32
    x = 4
    y = 4

    attributes = {}
    spectrum = Lognormal(n_part, v_mean, d)
    attributes["volume"], attributes["multiplicity"] = Linear(spectrum).sample(n_sd)
    particulator = DummyParticulator(backend_class, n_sd, grid=(x, y))
    particulator.croupier = croupier

    attributes["cell id"] = np.array((n_sd,), dtype=int)
    cell_origin_np = np.concatenate(
        [np.random.randint(0, x, n_sd), np.random.randint(0, y, n_sd)]
    ).reshape((2, -1))
    attributes["cell origin"] = cell_origin_np
    position_in_cell_np = np.concatenate(
        [np.random.rand(n_sd), np.random.rand(n_sd)]
    ).reshape((2, -1))
    attributes["position in cell"] = position_in_cell_np
    particulator.build(attributes)

    # Act
    u01 = backend_class.Storage.from_ndarray(np.random.random(n_sd))
    particulator.attributes.permutation(u01, local=particulator.croupier == "local")
    _ = particulator.attributes.cell_start

    # Assert
    diff = np.diff(
        particulator.attributes["cell id"][
            particulator.attributes._ParticleAttributes__idx
        ]
    )
    assert (diff >= 0).all()
