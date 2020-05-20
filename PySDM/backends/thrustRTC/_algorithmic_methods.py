"""
Created at 10.12.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import ThrustRTC as trtc
from ._maths_methods import MathsMethods
from .nice_thrust import nice_thrust
from .conf import NICE_THRUST_FLAGS


class AlgorithmicMethods:

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def calculate_displacement(dim, scheme, displacement, courant, cell_origin, position_in_cell):
        dim = trtc.DVInt64(dim)
        idx_length = trtc.DVInt64(position_in_cell.shape[0])
        courant_length = trtc.DVInt64(courant.shape[0])
        loop = trtc.For(['dim', 'idx_length', 'displacement', 'courant', 'courant_length', 'cell_origin', 'position_in_cell'], "droplet", f'''
            // Arakawa-C grid
            int _l_0 = cell_origin[droplet + 0];
            int _l_1 = cell_origin[droplet + idx_length];
            int _l = _l_0 + _l_1 * courant_length;
            int _r_0 = cell_origin[droplet + 0] + 1 * (dim == 0);
            int _r_1 = cell_origin[droplet + idx_length] + 1 * (dim == 1);
            int _r = _r_0 + _r_1 * courant_length;
            int omega = position_in_cell[droplet + idx_length * dim];
            int c_r = courant[_r];
            int c_l = courant[_l];
            displacement[droplet, dim] = {scheme(None, None, None)}
            ''')
        loop.launch_n(displacement.shape[0], [dim, idx_length, displacement, courant, courant_length, cell_origin, position_in_cell])

    __coalescence_body = trtc.For(['n', 'volume', 'idx', 'idx_length', 'intensive', 'intensive_length', 'extensive', 'extensive_length', 'gamma', 'healthy'], "i", '''
        if (gamma[i] == 0)
            return;

        int j = idx[i];
        int k = idx[i + 1];

        if (n[j] < n[k]) {
            j = idx[i + 1];
            k = idx[i];
        }
        int g = (int)(n[j] / n[k]);
        if (g > gamma[i])
            g = gamma[i];
        if (g == 0)
            return;
            
        int new_n = n[j] - g * n[k];
        
        if (new_n > 0) {
            n[j] = new_n;
            
            for (int attr = 0; attr < intensive_length; attr+=idx_length) {
                intensive[attr + k] = (intensive[attr + k] * volume[k] + intensive[attr + j] * g * volume[j]) / (volume[k] + g * volume[j]);
            }
            for (int attr = 0; attr < extensive_length; attr+=idx_length) {
                extensive[attr + k] += g * extensive[attr + j];
            }
        }
        else {  // new_n == 0
            n[j] = (int)(n[k] / 2);
            n[k] = n[k] - n[j];
            for (int attr = 0; attr < intensive_length; attr+=idx_length) {
                intensive[attr + j] = (intensive[attr + k] * volume[k] + intensive[attr + j] * g * volume[j]) / (volume[k] + g * volume[j]);
                intensive[attr + k] = intensive[attr + j];
            }
            for (int attr = 0; attr < extensive_length; attr+=idx_length) {
                extensive[attr + j] = g * extensive[attr + j] + extensive[attr + k];
                extensive[attr + k] = extensive[attr + j];
            }
        }
        if (n[k] == 0 || n[j] == 0) {
            healthy[0] = 0;
        }
        ''')

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def coalescence(n, volume, idx, length, intensive, extensive, gamma, healthy):
        idx_length = trtc.DVInt64(idx.size())
        intensive_length = trtc.DVInt64(intensive.size())
        extensive_length = trtc.DVInt64(extensive.size())
        AlgorithmicMethods.__coalescence_body.launch_n(length - 1, [n, volume, idx, idx_length, intensive, intensive_length, extensive, extensive_length, gamma, healthy])

    __compute_gamma_body = trtc.For(['prob', 'rand'], "i", '''
        prob[i] += rand[int(i / 2)];
        ''')

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def compute_gamma(prob, rand):
        MathsMethods.multiply(prob, -1.)
        AlgorithmicMethods.__compute_gamma_body.launch_n(prob.size(), [prob, rand])
        MathsMethods.floor(prob)
        MathsMethods.multiply(prob, -1.)

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def condensation(
            solver,
            n_cell, cell_start_arg,
            v, particle_temperatures, n, vdry, idx, rhod, thd, qv, dv, prhod, pthd, pqv, kappa,
            rtol_x, rtol_thd, dt, substeps, cell_order
    ):
        raise NotImplementedError()

    __flag_precipitated_body = trtc.For(['idx', 'idx_length', 'n_dims', 'healthy', 'cell_origin', 'position_in_cell'], "i", '''
        if (cell_origin[idx_length * (n_dims-1) + i] == 0 && position_in_cell[idx_length * (n_dims-1) + i] < 0) {
            idx[i] = idx_length;
            healthy[0] = 0;
        }
        ''')

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def flag_precipitated(cell_origin, position_in_cell, idx, length, healthy):
        idx_length = trtc.DVInt64(idx.size())
        n_dims = len(cell_origin.shape)
        AlgorithmicMethods.__flag_precipitated_body.launch_n(length, [idx, idx_length, n_dims, healthy, cell_origin, position_in_cell])

    @staticmethod
    def make_cell_caretaker(idx, cell_start, scheme):
        return AlgorithmicMethods._sort_by_cell_id_and_update_cell_start

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def moments(moment_0, moments, n, attr, cell_id, idx, length, specs_idx, specs_rank, min_x, max_x, x_id):
        # TODO print("Numba import!: ThrustRTC.moments(...)")

        from PySDM.backends.numba.numba import Numba
        from PySDM.backends.thrustRTC._storage_methods import StorageMethods
        host_moment_0 = StorageMethods.to_ndarray(moment_0)
        host_moments = StorageMethods.to_ndarray(moments)
        host_n = StorageMethods.to_ndarray(n)
        host_attr = StorageMethods.to_ndarray(attr)
        host_cell_id = StorageMethods.to_ndarray(cell_id)
        host_idx = StorageMethods.to_ndarray(idx)
        host_specs_idx = StorageMethods.to_ndarray(specs_idx)
        host_specs_rank = StorageMethods.to_ndarray(specs_rank)
        Numba.moments(host_moment_0, host_moments, host_n, host_attr, host_cell_id, host_idx, length,
                      host_specs_idx, host_specs_rank, min_x, max_x, x_id)
        device_moment_0 = StorageMethods.from_ndarray(host_moment_0)
        device_moments = StorageMethods.from_ndarray(host_moments)
        trtc.Copy(device_moment_0, moment_0)
        trtc.Copy(device_moments, moments)

    __normalize_body_0 = trtc.For(['cell_start', 'norm_factor', 'dt_div_dv'], "i", '''
        int sd_num = cell_start[i + 1] - cell_start[i];
        if (sd_num < 2) {
            norm_factor[i] = 0;
        }
        else {
            int half_sd_num = sd_num / 2;
            norm_factor[i] = dt_div_dv * sd_num * (sd_num - 1) / 2 / half_sd_num;
        }
        ''')

    __normalize_body_1 = trtc.For(['prob', 'cell_id', 'norm_factor'], "d", '''
        prob[d] *= norm_factor[cell_id[d]];
        ''')

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def normalize(prob, cell_id, cell_start, norm_factor, dt_div_dv):
        n_cell = cell_start.shape[0] - 1
        device_dt_div_dv = trtc.DVDouble(dt_div_dv)
        AlgorithmicMethods.__normalize_body_0.launch_n(n_cell, [cell_start, norm_factor, device_dt_div_dv])
        AlgorithmicMethods.__normalize_body_1.launch_n(prob.shape[0], [prob, cell_id, norm_factor])

    __remove_zeros_body = trtc.For(['data', 'idx', 'idx_length'], "i", '''
        if (idx[i] < idx_length && data[idx[i]] == 0)
            idx[i] = idx_length;
        ''')

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def remove_zeros(data, idx, length) -> int:
        idx_length = trtc.DVInt64(idx.size())

        # Warning: (potential bug source): reading from outside of array
        AlgorithmicMethods.__remove_zeros_body.launch_n(length, [data, idx, idx_length])

        trtc.Sort(idx)

        result = trtc.Find(idx, idx_length)
        if result is None:
            result = length

        return result

    ___sort_by_cell_id_and_update_cell_start_body = trtc.For(['cell_id', 'cell_start', 'idx'], "i", '''
        if (i == 0) {
            cell_start[cell_id[idx[0]]] = 0;
        } 
        else {
            int cell_id_curr = cell_id[idx[i]];
            int cell_id_next = cell_id[idx[i + 1]];
            int diff = (cell_id_next - cell_id_curr);
            for (int j = 1; j <= diff; j++) {
                cell_start[cell_id_curr + j] = idx[i + 1];
            }
        }
        ''')

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def _sort_by_cell_id_and_update_cell_start(cell_id, cell_start, idx, length):
        trtc.Sort_By_Key(idx.range(0, length), cell_id.range(0, length))
        trtc.Fill(cell_start, trtc.DVInt64(length))
        AlgorithmicMethods.___sort_by_cell_id_and_update_cell_start_body.launch_n(length - 1, [cell_id, cell_start, idx])
        return idx
