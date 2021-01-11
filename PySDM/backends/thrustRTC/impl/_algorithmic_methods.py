"""
Created at 10.12.2019
"""

from functools import reduce
from PySDM.backends.thrustRTC.conf import NICE_THRUST_FLAGS
from PySDM.backends.thrustRTC.nice_thrust import nice_thrust
from ..conf import trtc
from .precision_resolver import PrecisionResolver


class AlgorithmicMethods:

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def calculate_displacement(dim, scheme, displacement, courant, cell_origin, position_in_cell):
        dim = trtc.DVInt64(dim)
        idx_length = trtc.DVInt64(position_in_cell.shape[1])
        courant_length = trtc.DVInt64(courant.shape[0])
        loop = trtc.For(['dim', 'idx_length', 'displacement', 'courant', 'courant_length', 'cell_origin', 'position_in_cell'], "i", f'''
            // Arakawa-C grid
            auto _l_0 = cell_origin[i + 0];
            auto _l_1 = cell_origin[i + idx_length];
            auto _l = _l_0 + _l_1 * courant_length;
            auto _r_0 = cell_origin[i + 0] + 1 * (dim == 0);
            auto _r_1 = cell_origin[i + idx_length] + 1 * (dim == 1);
            auto _r = _r_0 + _r_1 * courant_length;
            auto omega = position_in_cell[i + idx_length * dim];
            auto c_r = courant[_r];
            auto c_l = courant[_l];
            displacement[i, dim] = {scheme(None, None, None)}
            ''')
        loop.launch_n(
            displacement.shape[1],
            [dim, idx_length, displacement.data, courant.data, courant_length, cell_origin.data, position_in_cell.data])

    __coalescence_body = trtc.For(['n', 'volume', 'idx', 'idx_length', 'intensive', 'intensive_length', 'extensive', 'extensive_length', 'gamma', 'healthy', 'adaptive', 'subs', 'adaptive_memory'], "i", '''
        if (gamma[i] == 0) {
            if (adaptive) {
                adaptive_memory[i] = 1;
            }
            return;
        }

        auto j = idx[i];
        auto k = idx[i + 1];

        if (n[j] < n[k]) {
            j = idx[i + 1];
            k = idx[i];
        }
        auto g = (int64_t)(n[j] / n[k]);
        if (adaptive) {
            adaptive_memory[i] = (int64_t)(gamma[i] * subs / g);
        }
        if (g > gamma[i]) {
            g = gamma[i];
        }
        if (g == 0) {
            return;
        }
            
        auto new_n = n[j] - g * n[k];
        if (new_n > 0) {
            n[j] = new_n;
            
            for (auto attr = 0; attr < intensive_length; attr+=idx_length) {
                intensive[attr + k] = (intensive[attr + k] * volume[k] + intensive[attr + j] * g * volume[j]) / (volume[k] + g * volume[j]);
            }
            for (auto attr = 0; attr < extensive_length; attr+=idx_length) {
                extensive[attr + k] += g * extensive[attr + j];
            }
        }
        else {  // new_n == 0
            n[j] = (int64_t)(n[k] / 2);
            n[k] = n[k] - n[j];
            for (auto attr = 0; attr < intensive_length; attr+=idx_length) {
                intensive[attr + j] = (intensive[attr + k] * volume[k] + intensive[attr + j] * g * volume[j]) / (volume[k] + g * volume[j]);
                intensive[attr + k] = intensive[attr + j];
            }
            for (auto attr = 0; attr < extensive_length; attr+=idx_length) {
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
    def coalescence(n, volume, idx, length, intensive, extensive, gamma, healthy, adaptive, cell_id, cell_idx, subs, adaptive_memory, collision_rate, collision_rate_deficit):
        idx_length = trtc.DVInt64(len(idx))
        intensive_length = trtc.DVInt64(len(intensive))
        extensive_length = trtc.DVInt64(len(extensive))
        adaptive_device = trtc.DVBool(adaptive)
        subs_device = trtc.DVInt64(subs[0])  # TODO #330
        AlgorithmicMethods.__coalescence_body.launch_n(length - 1,
            [n.data, volume.data, idx.data, idx_length, intensive.data, intensive_length, extensive.data, extensive_length, gamma.data, healthy.data, adaptive_device, subs_device, adaptive_memory.data])
        if adaptive:
            return trtc.Reduce(adaptive_memory.data.range(0, length-1), trtc.DVInt64(0), trtc.Maximum())
        else:
            return 1

    __compute_gamma_body = trtc.For(['prob', 'rand'], "i", '''
        prob[i] = ceil(prob[i] - rand[(int64_t)(i / 2)]);
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

    __flag_precipitated_body = trtc.For(['idx', 'idx_length', 'n_dims', 'healthy', 'cell_origin', 'position_in_cell',
                                         'volume', 'n'], "i", '''
        if (cell_origin[idx_length * (n_dims-1) + i] == 0 && position_in_cell[idx_length * (n_dims-1) + i] < 0) {
            idx[i] = idx_length;
            healthy[0] = 0;
        }
        ''')

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def flag_precipitated(cell_origin, position_in_cell, volume, n, idx, length, healthy):
        idx_length = trtc.DVInt64(len(idx))
        n_dims = trtc.DVInt64(len(cell_origin.shape))
        AlgorithmicMethods.__flag_precipitated_body.launch_n(
            length, [idx.data, idx_length, n_dims, healthy.data, cell_origin.data, position_in_cell.data,
                     volume.data, n.data])
        return 0  # TODO #332

    __linear_collection_efficiency_body = trtc.For(['A', 'B', 'D1', 'D2', 'E1', 'E2', 'F1', 'F2', 'G1', 'G2', 'G3', 'Mf', 'Mg', 'output', 'radii', 'is_first_in_pair', 'unit'], "i", '''
        output[i] = 0;
        if (is_first_in_pair[i]) {
            real_type r = radii[i] / unit;
            real_type r_s = radii[i + 1] / unit;
            real_type p = r_s / r;
            if (p != 0 && p != 1) {
                real_type G = pow((G1 / r), Mg) + G2 + G3 * r;
                real_type Gp = pow((1 - p), G);
                if (Gp != 0) {
                    real_type D = D1 / pow(r, D2);
                    real_type E = E1 / pow(r, E2);
                    real_type F = pow((F1 / r), Mf) + F2;
                    output[i] = A + B * p + D / pow(p, F) + E / Gp;
                    if (output[i] < 0) {
                        output[i] = 0;
                    }
                }
            }
        }
    '''.replace("real_type", PrecisionResolver.get_C_type()))

    @staticmethod
    def linear_collection_efficiency(params, output, radii, is_first_in_pair, unit):
        A, B, D1, D2, E1, E2, F1, F2, G1, G2, G3, Mf, Mg = params
        dA = PrecisionResolver.get_floating_point(A)
        dB = PrecisionResolver.get_floating_point(B)
        dD1 = PrecisionResolver.get_floating_point(D1)
        dD2 = PrecisionResolver.get_floating_point(D2)
        dE1 = PrecisionResolver.get_floating_point(E1)
        dE2 = PrecisionResolver.get_floating_point(E2)
        dF1 = PrecisionResolver.get_floating_point(F1)
        dF2 = PrecisionResolver.get_floating_point(F2)
        dG1 = PrecisionResolver.get_floating_point(G1)
        dG2 = PrecisionResolver.get_floating_point(G2)
        dG3 = PrecisionResolver.get_floating_point(G3)
        dMf = PrecisionResolver.get_floating_point(Mf)
        dMg = PrecisionResolver.get_floating_point(Mg)
        dunit = PrecisionResolver.get_floating_point(unit)
        AlgorithmicMethods.__linear_collection_efficiency_body.launch_n(len(is_first_in_pair) - 1,
            [dA, dB, dD1, dD2, dE1, dE2, dF1, dF2, dG1, dG2, dG3, dMf, dMg, output.data, radii.data, is_first_in_pair.indicator.data, dunit])

    __interpolation_body = trtc.For(['output', 'radius', 'factor', 'a', 'b'], 'i', '''
        auto r_id = (int64_t)(factor * radius[i]);
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
    def make_cell_caretaker(idx, cell_start, scheme=None):
        return AlgorithmicMethods._sort_by_cell_id_and_update_cell_start

    __loop_reset = trtc.For(['vector_to_clean'], "i",
    '''
        vector_to_clean[i] = 0;
    ''')

    __moments_body_0 = trtc.For(['idx', 'min_x', 'attr', 'x_id', 'max_x', 'moment_0', 'cell_id', 'n', 'specs_idx_shape', 'moments',
                                 'specs_idx', 'specs_rank', 'attr_shape', 'moments_shape'], "fake_i",
    '''
        auto i = idx[fake_i];
        if (min_x < attr[attr_shape * x_id + i] && attr[attr_shape  * x_id + i] < max_x) {
            atomicAdd((unsigned long long int*)&moment_0[cell_id[i]], (unsigned long long int)n[i]);
            for (auto k = 0; k < specs_idx_shape; k+=1) {
                atomicAdd((real_type*) &moments[moments_shape * k + cell_id[i]], n[i] * pow((real_type)(attr[attr_shape * specs_idx[k] + i]), (real_type)(specs_rank[k])));
            }
        }
    '''.replace("real_type", PrecisionResolver.get_C_type()))

    __moments_body_1 = trtc.For(['specs_idx_shape', 'moments', 'moment_0', 'moments_shape'], "c_id",
    '''
        for (auto k = 0; k < specs_idx_shape; k+=1) {
            if (moment_0[c_id] == 0) {
                moments[moments_shape * k  + c_id] = 0;
            } 
            else {
                moments[moments_shape * k + c_id] = moments[moments_shape * k + c_id] / moment_0[c_id];
            }
        }
    ''')

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def moments(moment_0, moments, n, attr, cell_id, idx, length, specs_idx, specs_rank, min_x, max_x, x_id):
        AlgorithmicMethods.__loop_reset.launch_n(moment_0.shape[0], [moment_0.data])
        AlgorithmicMethods.__loop_reset.launch_n(reduce(lambda x, y: x * y, moments.shape), [moments.data])

        AlgorithmicMethods.__moments_body_0.launch_n(length,
                           [idx.data, PrecisionResolver.get_floating_point(min_x), attr.data, trtc.DVInt64(x_id),
                            PrecisionResolver.get_floating_point(max_x),
                            moment_0.data, cell_id.data, n.data, trtc.DVInt64(specs_idx.shape[0]), moments.data,
                            specs_idx.data, specs_rank.data, trtc.DVInt64(attr.shape[1]),
                            trtc.DVInt64(moments.shape[1])])

        AlgorithmicMethods.__moments_body_1.launch_n(moment_0.shape[0], [trtc.DVInt64(specs_idx.shape[0]), moments.data, moment_0.data,
                           trtc.DVInt64(moments.shape[1])])


    __normalize_body_0 = trtc.For(['cell_start', 'norm_factor', 'dt_div_dv'], "i", '''
        auto sd_num = cell_start[i + 1] - cell_start[i];
        if (sd_num < 2) {
            norm_factor[i] = 0;
        }
        else {
            auto half_sd_num = sd_num / 2;
            norm_factor[i] = dt_div_dv * sd_num * (sd_num - 1) / 2 / half_sd_num;
        }
        ''')

    __normalize_body_1 = trtc.For(['prob', 'cell_id', 'norm_factor'], "i", '''
        prob[i] *= norm_factor[cell_id[i]];
        ''')

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def normalize(prob, cell_id, cell_idx, cell_start, norm_factor, dt, dv, n_substeps):  # TODO #69
        n_cell = cell_start.shape[0] - 1
        device_dt_div_dv = PrecisionResolver.get_floating_point(dt / dv)
        AlgorithmicMethods.__normalize_body_0.launch_n(n_cell, [cell_start.data, norm_factor.data, device_dt_div_dv])
        AlgorithmicMethods.__normalize_body_1.launch_n(prob.shape[0], [prob.data, cell_id.data, norm_factor.data])

    __remove_zeros_body = trtc.For(['data', 'idx', 'idx_length'], "i", '''
        if (idx[i] < idx_length && data[idx[i]] == 0) {
            idx[i] = idx_length;
        }
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
            auto cell_id_curr = cell_id[idx[i]];
            auto cell_id_next = cell_id[idx[i + 1]];
            auto diff = (cell_id_next - cell_id_curr);
            for (auto j = 1; j < diff + 1; j += 1) {
                cell_start[cell_id_curr + j] = idx[i + 1];
            }
        }
        ''')

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def _sort_by_cell_id_and_update_cell_start(cell_id, cell_idx, cell_start, idx, length):
        # TODO #69
        # if length > 0:
        #     max_cell_id = max(cell_id.to_ndarray())
        #     assert max_cell_id == 0
        trtc.Fill(cell_start.data, trtc.DVInt64(length))
        AlgorithmicMethods.___sort_by_cell_id_and_update_cell_start_body.launch_n(length - 1,
                                                                                  [cell_id.data, cell_start.data, idx.data])
        return idx
