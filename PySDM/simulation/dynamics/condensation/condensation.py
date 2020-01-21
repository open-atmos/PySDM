"""
Created at 24.10.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from .schemes.bdf import BDF
from .schemes.libcloud import ImplicitInSizeExplicitInTheta


class Condensation:
    def __init__(self, particles, kappa, scheme):

        self.particles = particles
        self.environment = particles.environment
        self.kappa = kappa

        mean_n_sd_in_cell = particles.n_sd//particles.mesh.n_cell
        if scheme == 'BDF':
            self.ode_solver = BDF(particles.backend, mean_n_sd_in_cell)
        elif scheme == 'libcloud':
            self.ode_solver = ImplicitInSizeExplicitInTheta(particles.backend, mean_n_sd_in_cell)
        else:
            raise NotImplementedError()

    def __call__(self):
        self.environment.sync()

        state = self.particles.state

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

            qv_new, thd_new = self.ode_solver.step(
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

            pqv[cell_id] = qv_new
            pthd[cell_id] = thd_new



