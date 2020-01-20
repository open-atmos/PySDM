"""
Created at 24.10.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from .schemes import BDF
from ...physics import formulae as phys
from ...physics import constants as const


class Condensation:
    def __init__(self, particles, kappa):

        self.particles = particles
        self.environment = particles.environment
        self.kappa = kappa
        self.scheme = 'scipy.odeint'
        self.ode_solver = BDF(particles.backend, particles.n_sd//particles.mesh.n_cell)

    def __call__(self):
        self.environment.sync()

        state = self.particles.state

        if self.scheme == 'scipy.odeint':
            for cell_id in range(self.particles.mesh.n_cell,):
                cell_start = state.cell_start[cell_id]
                cell_end = state.cell_start[cell_id + 1]
                n_sd_in_cell = cell_end - cell_start
                if n_sd_in_cell == 0:
                    continue

                prhod = self.environment.get_predicted("rhod")
                pthd = self.environment.get_predicted("thd")
                pqv = self.environment.get_predicted("qv")

                dt = self.particles.dt
                dthd_dt = (pthd[cell_id] - self.environment['thd'][cell_id]) / dt
                dqv_dt = (pqv[cell_id] - self.environment['qv'][cell_id]) / dt
                md_new = prhod[cell_id] * self.environment.dv
                md_old = self.environment['rhod'][cell_id] * self.environment.dv
                md_mean = (md_new + md_old) / 2
                rhod_mean = (prhod[cell_id] + self.environment['rhod'][cell_id]) / 2

                ml_new, ml_old, thd_new = self.ode_solver.step(
                    v=state.get_backend_storage("volume"),
                    n=state.n,
                    vdry=state.get_backend_storage("dry volume"),
                    cell_idx=self.particles.state._State__idx[cell_start:cell_end],
                    dt=self.particles.dt,
                    kappa=self.kappa,
                    thd=self.environment['thd'][cell_id],
                    qv=self.environment['qv'][cell_id],
                    dthd_dt=dthd_dt,
                    dqv_dt=dqv_dt,
                    m_d_mean=md_mean,
                    rhod_mean=rhod_mean
                )

                pqv[cell_id] -= (ml_new - ml_old) / md_mean

                env_old = self.environment
                exner_old = env_old['thd'][cell_id] / env_old['T'][cell_id]
                heat_old = phys.heat(T=env_old['T'][cell_id], qv=env_old['qv'][cell_id], ql=ml_old/md_mean)
                heat_new_over_T_new = phys.heat(T=1., qv=pqv[cell_id], ql=ml_new/md_mean)
                T_new = heat_old / heat_new_over_T_new
                pthd[cell_id] = exner_old * T_new

                # TODO
                # print(env_old['t'][cell_id])
                # import numpy as np
                # np.testing.assert_almost_equal(pthd[cell_id], thd_new, decimal=1)

        else:
            raise NotImplementedError()



