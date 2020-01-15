"""
Created at 24.10.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from .schemes import BDF


class Condensation:
    def __init__(self, particles, kappa):

        self.particles = particles
        self.environment = particles.environment
        self.kappa = kappa
        self.scheme = 'scipy.odeint'

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
                m_d_mean = (prhod[cell_id] + self.environment['rhod'][cell_id])/2 * self.environment.dv

                dm, thd_new = BDF.step(
                    v=state.get_backend_storage("volume"),
                    n=state.n,
                    vdry=state.get_backend_storage("dry volume"),
                    cell_idx=self.particles.state.idx[cell_start:cell_end],
                    dt=self.particles.dt,
                    kappa=self.kappa,
                    rhod=self.environment['rhod'][cell_id],
                    thd=self.environment['thd'][cell_id],
                    qv=self.environment['qv'][cell_id],
                    drhod_dt=drhod_dt,
                    dthd_dt=dthd_dt,
                    dqv_dt=dqv_dt,
                    m_d_mean=m_d_mean
                )

                pqv[cell_id] -= dm / m_d_mean
                pthd[cell_id] = thd_new  # TODO: same logic as above with tests for conservation of energy and mass
        else:
            raise NotImplementedError()



