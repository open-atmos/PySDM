"""
Created at 24.10.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from .schemes import bdf
from .schemes import libcloud
import numba
from ....backends.numba import conf
import numpy as np


default_rtol_lnv = 1e-9
default_rtol_thd = 1e-9


class Condensation:
    def __init__(self, particles, kappa, scheme,
                 rtol_lnv=default_rtol_lnv,
                 rtol_thd=default_rtol_thd,
                 ):
        self.particles = particles
        self.environment = particles.environment
        self.kappa = kappa
        self.rtol_lnv = rtol_lnv
        self.rtol_thd = rtol_thd

        self.scheme = scheme

        self.substeps = particles.backend.array(particles.mesh.n_cell, dtype=int)
        self.substeps[:] = np.maximum(1, int(particles.dt))

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
        jit_flags['parallel'] = thread_safe and particles.mesh.n_cell > 1
        jit_flags['nopython'] = thread_safe

        @numba.jit(**jit_flags)
        def condensation_step(
            impl, n_threads, n_cell, cell_start_arg,
            y, v, n, vdry, idx, rhod, thd, qv, dv, prhod, pthd, pqv, kappa,
            rtol_lnv, rtol_thd, dt, substeps, cell_order
        ):
            for thread_id in numba.prange(n_threads):
                for i in range(thread_id, n_cell, n_threads): # TODO: at least show that it is not slower :)
                    cell_id = cell_order[i]

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

                    qv_new, thd_new, substeps_hint = impl(
                        y, v, n, vdry,
                        idx[cell_start:cell_end], # TODO
                        kappa, thd[cell_id], qv[cell_id], dthd_dt, dqv_dt, md_mean, rhod_mean,
                        rtol_lnv, rtol_thd, dt, substeps[cell_id]
                    )

                    substeps[cell_id] = substeps_hint

                    pqv[cell_id] = qv_new
                    pthd[cell_id] = thd_new
        self.__condensation_step = condensation_step

    def __call__(self):
        self.environment.sync()
        self.__condensation_step(
            impl=self.impl,
            n_threads=self.particles.backend.num_threads(),
            n_cell=self.particles.mesh.n_cell,
            cell_start_arg=self.particles.state.cell_start,
            y=self.y,
            v=self.particles.state.get_backend_storage("volume"),
            n=self.particles.state.n,
            vdry=self.particles.state.get_backend_storage("dry volume"),
            idx=self.particles.state._State__idx,
            rhod=self.environment["rhod"],
            thd=self.environment["thd"],
            qv=self.environment["qv"],
            dv=self.environment.dv,
            prhod=self.environment.get_predicted("rhod"),
            pthd=self.environment.get_predicted("thd"),
            pqv=self.environment.get_predicted("qv"),
            kappa=self.kappa,
            rtol_lnv=self.rtol_lnv,
            rtol_thd=self.rtol_thd,
            dt=self.particles.dt,
            substeps=self.substeps,
            cell_order=np.argsort(self.substeps)
        )
