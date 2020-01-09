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
        self.cell_start = particles.backend.array(self.particles.mesh.n_cell + 1, dtype=int)
        self.scheme = 'scipy.odeint'  # TODO

    # TODO: assumes sorted by cell_id (i.e., executed after coalescence)
    def __call__(self):
        self.environment.sync()

        state = self.particles.state
        state.sort_by_cell_id(self.cell_start)  # TODO +what about droplets that precipitated out of the domain

        if self.scheme == 'scipy.odeint':
            for cell_id in range(self.particles.mesh.n_cell,):
                cell_start = self.cell_start[cell_id]
                cell_end = self.cell_start[cell_id + 1]
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
                m_d_mean = (prhod[cell_id] + self.environment['rhod'][cell_id])/2 * self.particles.mesh.dv

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



