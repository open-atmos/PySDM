"""
Created at 24.10.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from .schemes import bdf
from .schemes import libcloud
import numba
from ....backends.numba import conf


default_rtol = 1e-3
default_atol = 1e-3


class Condensation:
    def __init__(self, particles, kappa, scheme, rtol=default_rtol, atol=default_atol):

        self.particles = particles
        self.environment = particles.environment
        self.kappa = kappa
        self.rtol = rtol
        self.atol = atol

        self.scheme = scheme

        if scheme == 'BDF':
            mean_n_sd_in_cell = particles.n_sd // particles.mesh.n_cell
            self.y = particles.backend.array(2 * mean_n_sd_in_cell + bdf.idx_lnv, dtype=float)
            thread_safe = False
            self.impl = bdf.impl
        elif scheme == 'libcloud':  # TODO: rename!!!
            self.y = None
            thread_safe = True
            self.impl = libcloud.impl
        else:
            raise NotImplementedError()

        # TODO: move to backend
        jit_flags = conf.JIT_FLAGS.copy()
        jit_flags['parallel'] = thread_safe
        jit_flags['nopython'] = thread_safe
        @numba.jit(**jit_flags)
        def step(y, impl, n_cell, cell_start_arg, n, v, vdry, idx,
                                  dt, rhod, thd, qv, dv, prhod, pthd, pqv,
                                  kappa, rtol, atol):
            for cell_id in numba.prange(n_cell):  # TODO: after making dt automatic and f(S, N), consider more clever pool logic
                cell_start = cell_start_arg[cell_id]
                cell_end = cell_start_arg[cell_id + 1]
                n_sd_in_cell = cell_end - cell_start
                if n_sd_in_cell == 0:
                    continue

                dthd_dt = (pthd[cell_id] - thd[cell_id]) / dt
                dqv_dt = (pqv[cell_id] - qv[cell_id]) / dt
                md_new = prhod[cell_id] * dv
                md_old = rhod[cell_id] * dv
                md_mean = (md_new + md_old) / 2
                rhod_mean = (prhod[cell_id] + rhod[cell_id]) / 2

                qv_new, thd_new = impl(
                    y=y,
                    v=v,
                    n=n,
                    vdry=vdry,
                    cell_idx=idx[cell_start:cell_end],
                    dt=dt,
                    kappa=kappa,
                    thd=thd[cell_id],
                    qv=qv[cell_id],
                    dthd_dt=dthd_dt,
                    dqv_dt=dqv_dt,
                    m_d_mean=md_mean,
                    rhod_mean=rhod_mean,
                    rtol=rtol,
                    atol=atol
                )

                pqv[cell_id] = qv_new
                pthd[cell_id] = thd_new
        self.step = step

    def __call__(self):
        self.environment.sync()
        self.step(
            y=self.y,
            impl=self.impl,
            n_cell=self.particles.mesh.n_cell,
            cell_start_arg=self.particles.state.cell_start,
            n=self.particles.state.n,
            v=self.particles.state.get_backend_storage("volume"),
            vdry=self.particles.state.get_backend_storage("dry volume"),
            idx=self.particles.state._State__idx,
            dt=self.particles.dt,
            rhod=self.environment["rhod"],
            thd=self.environment["thd"],
            qv=self.environment["qv"],
            dv=self.environment.dv,
            prhod=self.environment.get_predicted("rhod"),
            pthd=self.environment.get_predicted("thd"),
            pqv=self.environment.get_predicted("qv"),
            kappa=self.kappa,
            rtol=self.rtol,
            atol=self.atol
        )
