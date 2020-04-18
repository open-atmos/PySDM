"""
Created at 10.12.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import ThrustRTC as trtc
from ._maths_methods import MathsMethods


class AlgorithmicMethods:

    @staticmethod
    def calculate_displacement(dim, scheme, displacement, courant, cell_origin, position_in_cell):
        idx_length = trtc.DVInt64(position_in_cell.shape[0])
        courant_length = trtc.DVInt64(courant.shape[0])
        loop = trtc.For(['dim', 'idx_length', 'displacement', 'courant', 'courant_length', 'cell_origin', 'position_in_cell'], "droplet", f'''
            // Arakawa-C grid
            _l_0 = cell_origin[droplet + 0];
            _l_1 = cell_origin[droplet + idx_length];
            _l = _l_0 + _l_1 * courant_length;
            _r_0 = cell_origin[droplet + 0] + 1 * (dim == 0);
            _r_1 = cell_origin[droplet + idx_length] + 1 * (dim == 1);
            _r = _r_0 + _r_1 * courant_length;
            omega = position_in_cell[droplet + idx_length * dim];
            c_r = courant[_r];
            c_l = courant[_l];
            displacement[droplet, dim] = {scheme(None, None, None)}
            ''')
        loop.launch_n(displacement.shape[0], [dim, idx_length, displacement, courant, courant_length, cell_origin, position_in_cell])

    @staticmethod
    def coalescence(n, volume, idx, length, intensive, extensive, gamma, healthy):
        loop = trtc.For(['n', 'volume', 'idx', 'intensive', 'extensive', 'gamma', 'healthy'], "i", '''
            auto j = 2 * i;
            auto k = j + 1;

            j = idx[j];
            k = idx[k];

            if (n[j] < n[k]) {
                auto old = j;
                j = k;
                k = old;
            }

            auto g = n[j] / n[k];
            if (g > gamma[i])
                g = gamma[i];

            if (g != 0) {
                auto new_n = n[j] - g * n[k];
                if (new_n > 0) {
                    n[j] = new_n;
                    intensive[/*:,*/ k] = (intensive[/*:,*/ k] * volume[k] + intensive[/*:,*/ j] * g * volume[j]) / (volume[k] + g * volume[j])
                    extensive[/*:,*/ k] += g * extensive[/*:,*/ j];
                }
                else {  // new_n == 0
                    n[j] = n[k] / 2;
                    n[k] = n[k] - n[j];
                    intensive[/*:,*/ k] = (intensive[/*:,*/ k] * volume[k] + intensive[/*:,*/ j] * g * volume[j]) / (volume[k] + g * volume[j])
                    intensive[/*:,*/ k] = intensive[/*:,*/ j]
                    extensive[/*:,*/ j] = g * extensive[/*:,*/ j] + extensive[/*:,*/ k];
                    extensive[/*:,*/ k] = extensive[/*:,*/ j];
                }
                if (n[j] == 0 || n[k] == 0) {
                    healthy[0] = 0;
                }
            }
            ''')
        loop.launch_n(length // 2, [n, volume, idx, intensive, extensive, gamma, healthy])

    @staticmethod
    def compute_gamma(prob, rand):
        MathsMethods.multiply(prob, -1.)
        loop = trtc.For(['prob', 'rand'], "i", "prob[i] += rand[int(i / 2)];")
        loop.launch_n(prob.size(), [prob, rand])
        MathsMethods.floor(prob)
        MathsMethods.multiply(prob, -1.)

    @staticmethod
    def condensation(
            solver,
            n_cell, cell_start_arg,
            v, particle_temperatures, n, vdry, idx, rhod, thd, qv, dv, prhod, pthd, pqv, kappa,
            rtol_x, rtol_thd, dt, substeps, cell_order
    ):
        raise NotImplementedError()

    @staticmethod
    def flag_precipitated(cell_origin, position_in_cell, idx, length, healthy):
        idx_length = trtc.DVInt64(idx.size())
        loop = trtc.For(['idx', 'idx_length', 'n_dims', 'healthy', 'cell_origin', 'position_in_cell'], "i", '''
                        if (cell_origin[idx_length * (n_dims-1) + i] == 0 && position_in_cell[idx_length * (n_dims-1) + i] < 0) {
                            idx[i] = idx_length;
                            healthy[0] = 0;
                        }
                        ''')
        n_dims = len(cell_origin.shape)
        loop.launch_n(length, [idx, idx_length, n_dims, healthy, cell_origin, position_in_cell])

    @staticmethod
    def make_cell_caretaker(_idx, _cell_start, _scheme):
        return AlgorithmicMethods._sort_by_cell_id_and_update_cell_start

    @staticmethod
    def moments(moment_0, moments, n, attr, cell_id, idx, length, specs_idx, specs_rank, min_x, max_x, x_id):
        raise NotImplementedError()

    @staticmethod
    def normalize(prob, cell_id, cell_start, norm_factor, dt_div_dv):
        raise NotImplementedError()

    @staticmethod
    def remove_zeros(data, idx, length) -> int:
        idx_length = trtc.DVInt64(idx.size())

        # Warning: (potential bug source): reading from outside of array
        loop = trtc.For(['data', 'idx', 'idx_length'], "i", '''
            if (data[idx[i]] == 0)
                idx[i] = idx_length;
            ''')
        loop.launch_n(length, [data, idx, idx_length])

        trtc.Sort(idx.range(0, length))

        result = trtc.Find(idx.range(0, length), idx_length)
        if result == -1:
            result = length

        return result

    @staticmethod
    def _sort_by_cell_id_and_update_cell_start(cell_id, cell_start, idx, length):
        trtc.Sort_By_Key(idx.range(0, length), cell_id.range(0, length))
        trtc.Fill(cell_start, length)
        loop = trtc.For(['cell_id', 'cell_start', 'idx'], "i", '''
            int cell_id_curr = cell_id[idx[i]];
            int cell_id_next = cell_id[idx[i + 1]];
            int diff = (cell_id_next - cell_id_curr);
            for (int j = 1; j <= diff; j++) {
                cell_start[cell_id_curr + j] = idx[i + 1];
            }
            ''')
        loop.launch_n(length - 1, [cell_id, cell_start, idx])

        return idx