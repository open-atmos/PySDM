#from ....backends_fixture import backend # TODO #588
from PySDM.backends import CPU
from PySDM import Builder
from PySDM.state.mesh import Mesh
from PySDM.dynamics.condensation import Condensation
import numpy as np
import re
import fnmatch


class _TestEnv:
    def __init__(self, dt, dv, rhod, thd, qv, T, p, RH):
        self.mesh = Mesh.mesh_0d()
        self.full = None
        self.particulator = None
        self.dt = dt
        self.dv = dv
        self.env = {'rhod': rhod, 'thd': thd, 'qv': qv, 'T': T, 'p': p, 'RH': RH}

    def register(self, builder):
        self.particulator = builder.particulator
        self.full = lambda item: self.particulator.backend.Storage.from_ndarray(np.full(self.particulator.n_sd, self.env[item]))

    def get_predicted(self, item):
        return self.full(item)

    def __getitem__(self, item):
        return self.full(item)


class _TestParticulator:
    def __init__(self, backend, n_sd=-1, max_iters=-1, multiplicity=-1,
                 dt=np.nan, dv=np.nan, rhod=np.nan, thd=np.nan, qv=np.nan, T=np.nan, p=np.nan, RH=np.nan,
                 dry_volume=np.nan, wet_radius=np.nan):
        builder = Builder(n_sd=n_sd, backend=backend())
        builder.set_environment(_TestEnv(dt=dt, dv=dv, rhod=rhod, thd=thd, qv=qv, T=T, p=p, RH=RH))
        builder.add_dynamic(Condensation(max_iters=max_iters))
        self.particulator = builder.build(attributes={
            'n': np.full(n_sd, multiplicity),
            'volume': np.full(n_sd, wet_radius),
            'dry volume': np.full(n_sd, dry_volume),
            'kappa times dry volume': np.ones(n_sd),
        })

    def run(self, steps):
        self.particulator.run(steps)


def _try(particulator, capsys):
    exception = None
    try:
        particulator.run(steps=1)
    except Exception as e:
       exception = e
    captured = capsys.readouterr()
    assert captured.out == ""
    assert isinstance(exception, RuntimeError)
    assert str(exception) == 'Condensation failed'
    return exception, captured.err


FN = fnmatch.translate("*condensation_methods.py")[:-2]


class TestDiagnostics:
    @staticmethod
    def test_burnout_long(capsys, backend=CPU):
        # arrange
        particulator = _TestParticulator(backend, dt=1, T=1, qv=1, dv=1, rhod=1, thd=1., max_iters=1, n_sd=1,
                                 multiplicity=1, dry_volume=1, wet_radius=1)

        # act
        exception, captured_err = _try(particulator, capsys)

        # assert
        pattern = re.compile(r"^burnout \(long\)\n\tfile: " + FN + r"\n\tcontext:\n\t\t thd\n\t\t 1.0\n$")
        assert pattern.match(captured_err) is not None
