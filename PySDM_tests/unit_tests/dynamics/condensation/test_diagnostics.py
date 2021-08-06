#from PySDM_tests.backends_fixture import backend # TODO #588
from PySDM.backends import CPU

from PySDM import Builder
from PySDM.environments import Parcel
from PySDM.state.mesh import Mesh
from PySDM.dynamics.condensation import Condensation
import numpy as np

class _TestEnv:
    def __init__(self, dt, dv):
        self.mesh = Mesh.mesh_0d()
        self.dt = dt
        self.dv = dv

    def register(self, builder):
        self.bck = builder.core.backend

    def get_predicted(self, item):
        return self.bck.Storage.from_ndarray(np.ones(1))

    def __getitem__(self, item):
        return self.bck.Storage.from_ndarray(np.ones(1))

class TestDiagnostics:
    @staticmethod
    def test_burnout_long(capsys, backend = CPU):
        # arrange
        n_sd = 1
        dt = 1
        dv = 1
        builder = Builder(n_sd=n_sd, backend=backend)
        builder.set_environment(_TestEnv(dt=dt, dv=dv))
        builder.add_dynamic(Condensation(kappa=0, max_iters=1))
        core = builder.build(attributes={'n': np.ones(n_sd), 'volume': np.ones(n_sd), 'dry volume': np.full(n_sd, 1)})

        # act
        exception = None
        try:
            core.run(steps=1)
        except Exception as e:
            exception = e
        captured = capsys.readouterr()

        # assert
        assert isinstance(exception, RuntimeError)
        assert captured.err == "condensation error: burnout (long)\ncontext:\n\t thd\n\t 1.0\n"
        assert captured.out == ""
