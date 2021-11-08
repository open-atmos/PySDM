from PySDM.backends.thrust_rtc.conf import NICE_THRUST_FLAGS
from PySDM.backends.thrust_rtc.impl import nice_thrust
from .precision_resolver import PrecisionResolver
from ..conf import trtc
from ...impl.backend_methods import BackendMethods


class DisplacementMethods(BackendMethods):
    def __init__(self):
        super().__init__()
        self.__calculate_displacement_body = trtc.For(
            (
                'dim', 'n_sd', 'displacement', 'courant', 'courant_length',
                'cell_origin', 'position_in_cell'
            ),
            "i",
            f'''
            // Arakawa-C grid
            auto _l_0 = cell_origin[i + 0];
            auto _l_1 = cell_origin[i + n_sd];
            auto _l = _l_0 + _l_1 * courant_length;
            auto _r_0 = cell_origin[i + 0] + 1 * (dim == 0);
            auto _r_1 = cell_origin[i + n_sd] + 1 * (dim == 1);
            auto _r = _r_0 + _r_1 * courant_length;
            auto omega = position_in_cell[i + n_sd * dim];
            auto c_r = courant[_r];
            auto c_l = courant[_l];
            displacement[i + n_sd * dim] = {
                self.formulae.particle_advection.displacement.c_inline(
                    c_l="c_l", c_r="c_r", omega="omega"
                )
            };
            '''.replace("real_type", PrecisionResolver.get_C_type()))

    @nice_thrust(**NICE_THRUST_FLAGS)
    def calculate_displacement(self, dim, displacement, courant, cell_origin, position_in_cell):
        dim = trtc.DVInt64(dim)
        n_sd = trtc.DVInt64(position_in_cell.shape[1])
        courant_length = trtc.DVInt64(courant.shape[0])
        self.__calculate_displacement_body.launch_n(
            displacement.shape[1],
            (
                dim, n_sd, displacement.data, courant.data, courant_length,
                cell_origin.data, position_in_cell.data
            )
        )

    __cell_id_body = trtc.For(('cell_id', 'cell_origin', 'strides', 'n_dims', 'size'), "i", '''
        cell_id[i] = 0;
        for (auto j = 0; j < n_dims; j += 1) {
            cell_id[i] += cell_origin[size * j + i] * strides[j];
        }
        ''')


    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def flag_precipitated(cell_origin, position_in_cell, volume, n, idx, length, healthy):
        n_sd = trtc.DVInt64(cell_origin.shape[1])
        n_dims = trtc.DVInt64(len(cell_origin.shape))
        rainfall = trtc.device_vector(PrecisionResolver.get_C_type(), 1)
        trtc.Fill(rainfall, PrecisionResolver.get_floating_point(0))
        AlgorithmicMethods.__flag_precipitated_body.launch_n(
            length, [idx.data, n_sd, n_dims, healthy.data, cell_origin.data, position_in_cell.data,
                     volume.data, n.data, rainfall])
        return rainfall.to_host()[0]

    __linear_collection_efficiency_body = trtc.For(
        ('A', 'B', 'D1', 'D2', 'E1', 'E2', 'F1', 'F2', 'G1', 'G2', 'G3', 'Mf', 'Mg',
         'output', 'radii', 'is_first_in_pair', 'idx', 'unit'),
        "i",
        '''
        if (is_first_in_pair[i]) {
            real_type r = 0;
            real_type r_s = 0;
            if (radii[idx[i]] > radii[idx[i + 1]]) {
                r = radii[idx[i]] / unit;
                r_s = radii[idx[i + 1]] / unit;
            }
            else {
                r = radii[idx[i + 1]] / unit;
                r_s = radii[idx[i]] / unit;
            }
            real_type p = r_s / r;
            if (p != 0 && p != 1) {
                real_type G = pow((G1 / r), Mg) + G2 + G3 * r;
                real_type Gp = pow((1 - p), G);
                if (Gp != 0) {
                    real_type D = D1 / pow(r, D2);
                    real_type E = E1 / pow(r, E2);
                    real_type F = pow((F1 / r), Mf) + F2;
                    output[int(i / 2)] = A + B * p + D / pow(p, F) + E / Gp;
                    if (output[int(i / 2)] < 0) {
                        output[int(i / 2)] = 0;
                    }
                }
            }
        }
    '''.replace("real_type", PrecisionResolver.get_C_type()))
