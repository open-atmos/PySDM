"""
Created at 10.12.2019
"""

import ThrustRTC as trtc
from PySDM.backends.thrustRTC.nice_thrust import nice_thrust
from PySDM.backends.thrustRTC.conf import NICE_THRUST_FLAGS


class AlgorithmicMethods:

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def calculate_displacement(dim, scheme, displacement, courant, cell_origin, position_in_cell):
        dim = trtc.DVInt64(dim)
        idx_length = trtc.DVInt64(position_in_cell.shape[1])
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
        loop.launch_n(displacement.shape[1], [dim, idx_length, displacement, courant, courant_length, cell_origin, position_in_cell])

    __coalescence_body = trtc.For(['n', 'volume', 'idx', 'idx_length', 'intensive', 'intensive_length', 'extensive', 'extensive_length', 'gamma', 'healthy', 'adaptive', 'subs', 'adaptive_memory'], "i", '''
        if (gamma[i] == 0) {
            adaptive_memory[i] = 1;
            return;
        }

        int j = idx[i];
        int k = idx[i + 1];

        if (n[j] < n[k]) {
            j = idx[i + 1];
            k = idx[i];
        }
        int g = n[j] / n[k];
        if (adaptive) 
            adaptive_memory[i] = (int)(gamma[i] * subs / g);
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
    def coalescence(n, volume, idx, length, intensive, extensive, gamma, healthy, adaptive, subs, adaptive_memory):
        idx_length = trtc.DVInt64(len(idx))
        intensive_length = trtc.DVInt64(len(intensive))
        extensive_length = trtc.DVInt64(len(extensive))
        adaptive_device = trtc.DVBool(adaptive)
        subs_device = trtc.DVInt64(subs)
        AlgorithmicMethods.__coalescence_body.launch_n(length - 1,
            [n.data, volume.data, idx.data, idx_length, intensive.data, intensive_length, extensive.data, extensive_length, gamma.data, healthy.data, adaptive_device, subs_device, adaptive_memory.data])
        return trtc.Reduce(adaptive_memory.data.range(0, length-1), trtc.DVInt64(0), trtc.Maximum())

    __compute_gamma_body = trtc.For(['prob', 'rand'], "i", '''
        prob[i] = -floor(-prob[i] + rand[int(i / 2)]);
        ''')

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def compute_gamma(prob, rand):
        AlgorithmicMethods.__compute_gamma_body.launch_n(len(prob), [prob.data, rand.data])

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
        n_dims = trtc.DVInt64(len(cell_origin.shape))
        AlgorithmicMethods.__flag_precipitated_body.launch_n(length, [idx, idx_length, n_dims, healthy, cell_origin, position_in_cell])

    __linear_collection_efficiency_body = trtc.For(['A', 'B', 'D1', 'D2', 'E1', 'E2', 'F1', 'F2', 'G1', 'G2', 'G3', 'Mf', 'Mg', 'output', 'radii', 'is_first_in_pair', 'unit'], "i", '''
        output[i] = 0;
        if (is_first_in_pair[i]) {
            double r = radii[i] / unit;
            double r_s = radii[i + 1] / unit;
            double p = r_s / r;
            if (p != 0 && p != 1) {
                double G = pow((G1 / r), Mg) + G2 + G3 * r;
                double Gp = pow((1 - p), G);
                if (Gp != 0) {
                    double D = D1 / pow(r, D2);
                    double E = E1 / pow(r, E2);
                    double F = pow((F1 / r), Mf) + F2;
                    output[i] = A + B * p + D / pow(p, F) + E / Gp;
                    if (output[i] < 0) {
                        output[i] = 0;
                    }
                }
            }
        }
    ''')

    @staticmethod
    def linear_collection_efficiency(params, output, radii, is_first_in_pair, unit):
        A, B, D1, D2, E1, E2, F1, F2, G1, G2, G3, Mf, Mg = params
        dA = trtc.DVDouble(A)
        dB = trtc.DVDouble(B)
        dD1 = trtc.DVDouble(D1)
        dD2 = trtc.DVDouble(D2)
        dE1 = trtc.DVDouble(E1)
        dE2 = trtc.DVDouble(E2)
        dF1 = trtc.DVDouble(F1)
        dF2 = trtc.DVDouble(F2)
        dG1 = trtc.DVDouble(G1)
        dG2 = trtc.DVDouble(G2)
        dG3 = trtc.DVDouble(G3)
        dMf = trtc.DVDouble(Mf)
        dMg = trtc.DVDouble(Mg)
        dunit = trtc.DVDouble(unit)
        AlgorithmicMethods.__linear_collection_efficiency_body.launch_n(len(is_first_in_pair) - 1,
            [dA, dB, dD1, dD2, dE1, dE2, dF1, dF2, dG1, dG2, dG3, dMf, dMg, output.data, radii.data, is_first_in_pair.data, dunit])

    __interpolation_body = trtc.For(['output', 'radius', 'factor', 'a', 'b'], 'i', '''
        int r_id = (int)(factor * radius[i]);
        auto r_rest = (factor * radius[i] - r_id) / factor;
        output[i] = a[r_id] + r_rest * b[r_id];
    ''')

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def interpolation(output, radius, factor, b, c):
        factor_device = trtc.DVInt64(factor)
        AlgorithmicMethods.__interpolation_body.launch_n(len(radius),
                                                         [output.data, radius.data, factor_device, b.data, c.data])

    @staticmethod
    def make_cell_caretaker(idx, cell_start, scheme):
        return AlgorithmicMethods._sort_by_cell_id_and_update_cell_start

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def moments(moment_0, moments, n, attr, cell_id, idx, length, specs_idx, specs_rank, min_x, max_x, x_id):
        # TODO print("Numba import!: ThrustRTC.moments(...)")

        from PySDM.backends.numba.numba import Numba
        host_moment_0 = moment_0.to_ndarray()
        host_moments = moments.to_ndarray()
        host_n = n.to_ndarray()
        host_attr = attr.to_ndarray()
        host_cell_id = cell_id.to_ndarray()
        host_idx = idx.to_ndarray()
        host_specs_idx = specs_idx.to_ndarray()
        host_specs_rank = specs_rank.to_ndarray()
        Numba.moments_body(host_moment_0, host_moments, host_n, host_attr, host_cell_id, host_idx, length,
                           host_specs_idx, host_specs_rank, min_x, max_x, x_id)
        moment_0.upload(host_moment_0)
        moments.upload(host_moments)

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
        AlgorithmicMethods.__normalize_body_0.launch_n(n_cell, [cell_start.data, norm_factor.data, device_dt_div_dv])
        AlgorithmicMethods.__normalize_body_1.launch_n(prob.shape[0], [prob.data, cell_id.data, norm_factor.data])

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

        result = idx.size() - trtc.Count(idx, idx_length)
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
        trtc.Sort_By_Key(cell_id.data, idx.data)
        trtc.Fill(cell_start.data, trtc.DVInt64(length))
        AlgorithmicMethods.___sort_by_cell_id_and_update_cell_start_body.launch_n(length - 1,
                                                                                  [cell_id.data, cell_start.data, idx.data])
        return idx
