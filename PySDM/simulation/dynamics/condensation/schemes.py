"""
Created at 09.01.2020

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""


import numpy as np
from scipy import integrate as ode
from ._odesystem import _ODESystem, idx_qv, idx_lnv, idx_rhod, idx_thd
from PySDM.simulation.physics.constants import rho_w


# class TODO:
#     @staticmethod
#     def step(**args):
#         np.save("C:\\Users\\piotr\\PycharmProjects\\PySDM\\PySDM_tests\\unit_tests\\simulation\\dynamics\\condensation\\test_data.npy", args)
#         return BDF.step(**args)


class Solver:
    def __init__(self, backend, mean_n_sd_in_cell):
        length = 2 * mean_n_sd_in_cell + 3
        self.y = backend.array(length, dtype=float)  # TODO: list

class Integ:
    pass

class EE(Solver):
    memory_requirement = 1

    def step(self,
             v, n, vdry,
             cell_idx,
             dt, kappa,
             rhod, thd, qv,
             drhod_dt, dthd_dt, dqv_dt,
             m_d_mean
        ):
        n_sd_in_cell = len(cell_idx)
        y0 = self.y[0:n_sd_in_cell + idx_lnv]
        y0[idx_rhod] = rhod
        y0[idx_thd] = thd
        y0[idx_qv] = qv

        y0[idx_lnv:] = np.log(v[cell_idx])  # TODO: abstract out ln()
        integ = self.solve_ivp(
            _ODESystem(
                kappa,
                vdry[cell_idx],
                n[cell_idx],
                drhod_dt,
                dthd_dt,
                dqv_dt,
                m_d_mean
            ),
            (0., dt),
            y0,
            rtol=1e-3,
            atol=1e-3,
            # first_step=self.dt,
        )

        m_new = 0
        m_old = 0
        for i in range(n_sd_in_cell):
            x_new = np.exp(integ.y[idx_lnv + i])
            x_old = v[cell_idx[i]]
            nd = n[cell_idx[i]]
            m_new += nd * x_new * rho_w
            m_old += nd * x_old * rho_w
            v[cell_idx[i]] = x_new

        return m_new, m_old, integ.y[idx_thd]

    def solve_ivp(self, odesys, t_range,
            y0,
            rtol=1e-3,
            atol=1e-3):
        integ = Integ()
        integ.y = y0
        dt = t_range[1] - t_range[0]
        integ.y += dt * odesys(None, integ.y)  # TODO: backend.add()
        return integ



class BDF(Solver):
    memory_requirement = 1

    def step(self,
             v, n, vdry,
             cell_idx,
             dt, kappa,
             rhod, thd, qv,
             drhod_dt, dthd_dt, dqv_dt,
             m_d_mean
        ):
        n_sd_in_cell = len(cell_idx)
        y0 = self.y[0:n_sd_in_cell + idx_lnv]
        y0[idx_rhod] = rhod
        y0[idx_thd] = thd
        y0[idx_qv] = qv
        y0[idx_lnv:] = np.log(v[cell_idx])  # TODO: abstract out ln()
        integ = ode.solve_ivp(
            _ODESystem(
                kappa,
                vdry[cell_idx],
                n[cell_idx],
                drhod_dt,
                dthd_dt,
                dqv_dt,
                m_d_mean
            ),
            (0., dt),
            y0,
            method='BDF',
            rtol=1e-3,
            atol=1e-3,
            # first_step=self.dt,
            t_eval=[dt]
        )
        assert integ.success, integ.message

        m_new = 0
        m_old = 0
        for i in range(n_sd_in_cell):
            x_new = np.exp(integ.y[idx_lnv + i])
            x_old = v[cell_idx[i]]
            nd = n[cell_idx[i]]
            m_new += nd * x_new * rho_w
            m_old += nd * x_old * rho_w
            v[cell_idx[i]] = x_new

        return m_new, m_old, integ.y[idx_thd]

