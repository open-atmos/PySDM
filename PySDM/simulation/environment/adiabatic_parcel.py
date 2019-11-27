

from PySDM.simulation.physics import formulae as phys
from PySDM.simulation.physics import constants as const
import numpy as np


class AdiabaticParcel: # TODO: inherit from environmrnt.moist_air!

    def __init__(self, particles, mass, p0, q0, T0, w):
        self.mass = mass
        self.w = w
        self.particles = particles

        self.t = 0.

        variables = ['qv', 'thd', 'T', 'p', 'RH', "rhod"]
        self._values = {
            "new": None,
            "old": self._allocate(variables)
        }
        self._tmp = self._allocate(variables)

        pd0 = p0 # TODO !!!!!!!!!!!!!!!!!!!!!
        initial_values = self._values["old"]
        initial_values["qv"][:] = q0
        initial_values["thd"][:] = phys.th_std(pd0, T0)
        initial_values["rhod"][:] = pd0 / const.Rd / T0

        self.qv_lambda = lambda: self._values["old"]["qv"]
        self.thd_lambda = lambda: self._values["old"]["thd"]

        self.sync()
        self._values["new"]["rhod"] = self._values["old"]["rhod"]

        np.testing.assert_approx_equal(self._values["new"]['T'][0], T0)
        # TODO: same for p (after fixing the above pd0 issue !!!!!)

        self._swap()

    @property
    def dv(self):
        raise NotImplementedError()

    @property
    def n_cell(self):
        return 1

    def __getitem__(self, index):
        values = self._values[index]
        if values is None:
            raise Exception("condensation not called.")
        return values

    def _allocate(self, variables):
        result = {}
        for var in variables:
            result[var] = self.particles.backend.array((self.n_cell,), float)
        return result

    @property
    def rhod(self):
        return self._values["old"]["rhod"]

    def sync(self):
        target = self._tmp
        self.particles.backend.upload(self.qv_lambda().ravel(), target['qv'])
        self.particles.backend.upload(self.thd_lambda().ravel(), target['thd'])

        self.particles.backend.apply(
            function=self.particles.backend.temperature_pressure_RH,
            args=(self.rhod, target['thd'], target['qv']),
            output=(target['T'], target['p'], target['RH'])
        )

        self._values["new"] = target

    def ante_step(self):
        dt = self.particles.dt
        old = self._values["old"]
        new = self._values["old"]

        # mid-point value for w (TODO?)
        self.t += self.particles.dt
        w = self.w(self.t - dt/2)

        pv = 0 # TODO!!!!!!!

        # Explicit Euler for p,T (predictor step)
        dpd_dt = - old["rhod"] * const.g * w
        dT_dt = dpd_dt / old["rhod"] / phys.c_p(old["qv"][0]) # TODO: consider true dT_dt(p, ...)
        new["rhod"][:] = old["rhod"] + dt * (
                dpd_dt / const.Rd / old["T"] +
                -dT_dt * (old["p"] - pv) / const.Rd / old["T"]**2
        )
        pass

    def post_step(self):
        self._swap()

    def _swap(self):
        self._tmp = self._values["old"]
        self._values["old"] = self._values["new"]
        self._values["new"] = None
