import numpy as np
import pytest
from PySDM.backends import ThrustRTC
from PySDM.builder import Builder
from PySDM.dynamics import Coalescence
from PySDM.environments import Box
from PySDM.initialisation.spectral_sampling import ConstantMultiplicity
from PySDM.physics.coalescence_kernels import Geometric, Electric, Hydrodynamic
from PySDM_examples.Berry_1967.settings import Settings
# noinspection PyUnresolvedReferences
from PySDM_tests.backends_fixture import backend


@pytest.mark.parametrize('croupier', ['local', 'global'])
@pytest.mark.parametrize('adaptive', [True, False])
@pytest.mark.parametrize('kernel', [Geometric(), Electric(), Hydrodynamic()])
def test_coalescence(backend, kernel, croupier, adaptive):
    if backend == ThrustRTC and croupier == 'local':  # TODO #358
        return
    if backend == ThrustRTC and adaptive and croupier == 'global':  # TODO #329
        return
    # Arrange
    s = Settings()
    s.formulae.seed = 0
    steps = [0, 200]

    builder = Builder(n_sd=s.n_sd, backend=backend, formulae=s.formulae)
    builder.set_environment(Box(dt=s.dt, dv=s.dv))
    attributes = {}
    attributes['volume'], attributes['n'] = ConstantMultiplicity(s.spectrum).sample(s.n_sd)
    builder.add_dynamic(Coalescence(kernel, croupier=croupier, adaptive=adaptive))
    core = builder.build(attributes)

    volumes = {}

    # Act
    for step in steps:
        core.run(step - core.n_steps)
        volumes[core.n_steps] = core.particles['volume'].to_ndarray()

    # Assert
    x_max = 0
    for volume in volumes.values():
        assert x_max < np.amax(volume)
        x_max = np.amax(volume)
