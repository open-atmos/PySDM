import numpy as np
from PySDM import Formulae
from PySDM.physics import si
from PySDM.dynamics.collisions.coalescence_efficiencies import ConstEc
from PySDM.initialisation.spectra import Gamma

class Settings:  # pylint: disable=too-many-instance-attributes,too-few-public-methods,missing-class-docstring
    def __init__(
        self,
        kernel: object,
        output_interval: float,
        n_part: float,
        radius_bins_edges: np.ndarray = None,
        n_sd = 2**12,
        dt = 1*si.s,
        L = 1e-6 * si.cm**3 / si.cm**3, # total volumetric liquid water content
        t_max = 1800 * si.s,
    ):
        self.kernel = kernel
        self.n_sd = n_sd
        self.n_part = n_part
        self.dv = 1 * si.m**3
        self.rho = 1000 * si.kg / si.m**3
        self.rhod = 1 * si.kg / si.m**3
        self.dt = dt
        self.adaptive = True
        self.output_interval = output_interval
        self.nt = int(t_max / self.dt)
        self.coal_eff = ConstEc(Ec=1.0)
        self.k = 2.0

        self.X0 = L / self.k / self.n_part
        self.spectrum = Gamma(
            norm_factor=self.n_part * self.dv, k=2.0, theta=self.X0
        )
        self.radius_bins_edges = radius_bins_edges if radius_bins_edges is not None else np.logspace(
            np.log10(8.0 * si.um), np.log10(5000 * si.um), num=64, endpoint=True
        )
        self.formulae = Formulae()

    @property
    def steps_per_output_interval(self) -> int:
        return int(self.output_interval / self.dt)