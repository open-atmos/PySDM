"""
Created at 04.11.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from PySDM.simulation.dynamics.coalescence import SDM
from PySDM.simulation.state import State
from PySDM.backends.default import Default
import numpy as np
from tests.unit_tests.simulation.test_state import TestState
from tests.unit_tests.simulation.dynamics.coalescence.__parametrisation__ import StubKernel, backend_fill

# noinspection PyUnresolvedReferences
from tests.unit_tests.simulation.dynamics.coalescence.__parametrisation__ import x_2, n_2


backend = Default()


class TestSDMMultiCell:
    def test_single_collision(self, n_2):
        # Arrange
        sut = SDM(StubKernel(), dt=0, dv=1, n_sd=len(n_2), n_cell=1, backend=backend)
        state = TestState.get_empty_state()
        state.n = backend.from_ndarray(n_2)
        state.idx = backend.from_ndarray(np.arange(2))
        state.SD_num = 2
        state.cell_id = backend.from_ndarray(np.arange(2))

        # Act
        sut(state)

        # Assert
        assert np.sum(state['n']) == np.sum(n_2) - np.amin(n_2)
        assert np.amax(state['n']) == max(np.amax(n_2) - np.amin(n_2), np.amin(n_2))
