from ...dummy_particulator import DummyParticulator
from PySDM.dynamics import Displacement
import numpy as np
from ...dummy_environment import DummyEnvironment
from PySDM.physics import Formulae


class DisplacementSettings:
    def __init__(self):
        self.n = np.ones(1, dtype=np.int64)
        self.volume = np.ones(1, dtype=np.float64)
        self.grid = (1, 1)
        self.courant_field_data = (np.array([[0, 0]]).T, np.array([[0, 0]]))
        self.positions = [[0], [0]]
        self.sedimentation = False
        self.dt = None

    def get_displacement(self, backend, scheme):
        formulae = Formulae(particle_advection=scheme)
        particulator = DummyParticulator(backend, n_sd=len(self.n), formulae=formulae)
        particulator.environment = DummyEnvironment(
            dt=self.dt,
            grid=self.grid,
            courant_field_data=self.courant_field_data)
        positions = np.array(self.positions)
        cell_id, cell_origin, position_in_cell = particulator.mesh.cellular_attributes(positions)
        attributes = {
            'n': self.n,
            'volume': self.volume,
            'cell id': cell_id,
            'cell origin': cell_origin,
            'position in cell': position_in_cell
        }
        particulator.build(attributes)
        sut = Displacement(enable_sedimentation=self.sedimentation)
        sut.register(particulator)
        sut.upload_courant_field(self.courant_field_data)

        return sut, particulator
