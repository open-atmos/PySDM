from PySDM.physics import constants as const


class CondensationMethods:
    def make_condensation_solver(self, dt, dt_range, adaptive):

        def calculate_ml_old(v, n, cell_idx):
            result = 0
            for drop in cell_idx:
                result += n[drop] * v[drop] * const.rho_w
            return result

        def step(args, dt, n_substep):
            step_iml(*args, dt, n_substep, fake=False)

        def step_iml(v, v_cr, n, vdry, cell_idx, kappa, thd, qv, dthd_dt, dqv_dt, m_d, rhod_mean,
                  rtol_x, rtol_thd, dt, n_substep, fake):
            ml_old = calculate_ml_old(v, n, cell_idx)

        def solve(v, v_cr, n, vdry, cell_idx, kappa, thd, qv, dthd_dt, dqv_dt, m_d, rhod_mean,
                  rtol_x, rtol_thd, dt, n_substeps):
            args = (v, v_cr, n, vdry, cell_idx, kappa, thd, qv, dthd_dt, dqv_dt, m_d, rhod_mean, rtol_x)

            step(args, dt, n_substeps)

        return solve