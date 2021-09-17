import numpy as np
import pytest

from PySDM.physics.spectra import Lognormal
from PySDM.initialisation.spectral_sampling import Linear
from ...dummy_environment import DummyEnvironment
from ...dummy_particulator import DummyParticulator

# noinspection PyUnresolvedReferences
from ....backends_fixture import backend


@pytest.mark.parametrize('croupier', ['local', 'global'])
def test_final_state(croupier, backend):
    from PySDM.backends import ThrustRTC
    if backend is ThrustRTC:
        return  # TODO #330

    # Arrange
    n_part = 100000
    v_mean = 2e-6
    d = 1.2
    n_sd = 32
    x = 4
    y = 4

    attributes = {}
    spectrum = Lognormal(n_part, v_mean, d)
    attributes['volume'], attributes['n'] = Linear(spectrum).sample(n_sd)
    particulator = DummyParticulator(backend, n_sd)
    particulator.environment = DummyEnvironment(grid=(x, y))
    particulator.croupier = croupier

    attributes['cell id'] = np.array((n_sd,), dtype=int)
    cell_origin_np = np.concatenate([np.random.randint(0, x, n_sd), np.random.randint(0, y, n_sd)]).reshape((2, -1))
    attributes['cell origin'] = cell_origin_np
    position_in_cell_np = np.concatenate([np.random.rand(n_sd), np.random.rand(n_sd)]).reshape((2, -1))
    attributes['position in cell'] = position_in_cell_np
    particulator.build(attributes)

    # Act
    u01 = backend.Storage.from_ndarray(np.random.random(n_sd))
    particulator.attributes.permutation(u01, local=particulator.croupier == 'local')
    _ = particulator.attributes.cell_start

    # Assert
    assert (np.diff(particulator.attributes['cell id'][particulator.attributes._Particles__idx]) >= 0).all()
