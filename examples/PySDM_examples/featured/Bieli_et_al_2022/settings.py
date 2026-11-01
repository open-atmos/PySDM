from pystrict import strict

from PySDM.dynamics.collisions.breakup_efficiencies import ConstEb
from PySDM.dynamics.collisions.breakup_fragmentations import Feingold1988
from PySDM.dynamics.collisions.coalescence_efficiencies import ConstEc
from PySDM.dynamics.collisions.collision_kernels import Golovin
from PySDM.formulae import Formulae
from PySDM.initialisation.spectra import Gamma
from PySDM.physics import si
from PySDM.physics.constants_defaults import rho_w


@strict
class Settings:
    def __init__(self, formulae: Formulae = None):
        self.n_sd = 2**12
        self.n_part = 1e4 / si.cm**3
        self.theta = 0.33e-9 * si.g / rho_w
        self.k = 1
        self.dv = 0.1 * si.m**3
        self.norm_factor = self.n_part * self.dv
        self.dt = 1 * si.seconds
        self.adaptive = False
        self.seed = 44
        self._steps = list(range(60))
        self.kernel = Golovin(b=2000 * si.cm**3 / si.g / si.s * rho_w)
        self.coal_effs = [ConstEc(Ec=0.8), ConstEc(Ec=0.9), ConstEc(Ec=1.0)]
        self.vmin = 1.0 * si.um**3
        self.nfmax = 10
        self.fragtol = 1e-3
        self.fragmentation = Feingold1988(
            scale=self.k * self.theta,
            fragtol=self.fragtol,
            vmin=self.vmin,
            nfmax=self.nfmax,
        )
        self.break_eff = ConstEb(1.0)
        self.spectrum = Gamma(norm_factor=self.norm_factor, k=self.k, theta=self.theta)
        self.rho = rho_w
        self.formulae = formulae or Formulae(
            fragmentation_function=self.fragmentation.__class__.__name__
        )

    @property
    def output_steps(self):
        return [int(step / self.dt) for step in self._steps]
