import numpy as np
import pytest
import struct
from PySDM.backends import ThrustRTC
from PySDM.builder import Builder
from PySDM.dynamics import Coalescence
from PySDM.environments import Box
from PySDM.initialisation.spectral_sampling import ConstantMultiplicity
from PySDM.physics.coalescence_kernels import Golovin, Geometric, Electric, Hydrodynamic
from PySDM_examples.Berry_1967.settings import Settings
# noinspection PyUnresolvedReferences
from ...backends_fixture import backend


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
    steps = [0, 800]

    builder = Builder(n_sd=s.n_sd, backend=backend(formulae=s.formulae))
    builder.set_environment(Box(dt=s.dt, dv=s.dv))
    attributes = {}
    attributes['volume'], attributes['n'] = ConstantMultiplicity(s.spectrum).sample(s.n_sd)
    builder.add_dynamic(Coalescence(kernel, croupier=croupier, adaptive=adaptive))
    particulator = builder.build(attributes)

    volumes = {}

    # Act
    for step in steps:
        particulator.run(step - particulator.n_steps)
        volumes[particulator.n_steps] = particulator.attributes['volume'].to_ndarray()

    # Assert
    x_max = 0
    for volume in volumes.values():
        assert x_max < np.amax(volume)
        x_max = np.amax(volume)


@pytest.mark.xfail(struct.calcsize("P") * 8 == 32, reason="32 bit", strict=False)
def test_coalescence_2_sd(backend):
    # Arrange
    s = Settings()
    s.kernel = Golovin(b=1.5e12)
    s.formulae.seed = 0
    steps = [0, 200]
    s.n_sd = 2

    builder = Builder(n_sd=s.n_sd, backend=backend(formulae=s.formulae))
    builder.set_environment(Box(dt=s.dt, dv=s.dv))
    attributes = {}
    attributes['volume'], attributes['n'] = ConstantMultiplicity(s.spectrum).sample(s.n_sd)
    builder.add_dynamic(Coalescence(s.kernel, adaptive=False))
    particulator = builder.build(attributes)

    volumes = {}

    # Act
    for step in steps:
        particulator.run(step - particulator.n_steps)
        volumes[particulator.n_steps] = particulator.attributes['volume'].to_ndarray()

    # Assert
    x_max = 0
    for volume in volumes.values():
        assert x_max < np.amax(volume)
        x_max = np.amax(volume)
    assert particulator.attributes.SD_num == 1
