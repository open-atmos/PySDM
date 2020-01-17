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
                drhod_dt = (prhod[cell_id] - self.environment['rhod'][cell_id]) / dt
                dthd_dt = (pthd[cell_id] - self.environment['thd'][cell_id]) / dt
                dqv_dt = (pqv[cell_id] - self.environment['qv'][cell_id]) / dt
                md_new = prhod[cell_id] * self.environment.dv
                md_old = self.environment['rhod'][cell_id] * self.environment.dv
                md_mean = (md_new + md_old) / 2

                ml_new, ml_old, thd_new = self.ode_solver.step(
                    v=state.get_backend_storage("volume"),
                    n=state.n,
                    vdry=state.get_backend_storage("dry volume"),
                    cell_idx=self.particles.state._State__idx[cell_start:cell_end],
                    dt=self.particles.dt,
                    kappa=self.kappa,
                    rhod=self.environment['rhod'][cell_id],
                    thd=self.environment['thd'][cell_id],
                    qv=self.environment['qv'][cell_id],
                    drhod_dt=drhod_dt,
                    dthd_dt=dthd_dt,
                    dqv_dt=dqv_dt,
                    m_d_mean=md_mean
                )

                pqv[cell_id] -= (ml_new - ml_old) / md_mean

                # old = self.environment
                # exner0 = old['thd'][cell_id] / old['T'][cell_id]
                # mse0 = phys.mse(T=old['T'][cell_id], qv=old['qv'][cell_id], ql=ml_old/md_old, z=0.)
                # mse1_over_T = phys.mse(T=1., qv=pqv[cell_id], ql=ml_new/md_new, z=0.)
                # if drhod_dt != 0:
                #     pass
                #     # dz = self.environment.get_predicted('z') - old['z'][cell_id]
                #     # mse1_over_T += (1 + old['qv'][cell_id]) * const.g * dz
                #     # # TODO!!!
                # T_new = mse0 / mse1_over_T
                # pthd[cell_id] = exner0 * T_new

                pthd[cell_id] = thd_new

        else:
            raise NotImplementedError()



