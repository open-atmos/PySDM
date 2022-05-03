"""
GPU implementation of backend methods for particle displacement (advection and sedimentation)
"""
from PySDM.backends.impl_thrust_rtc.conf import NICE_THRUST_FLAGS
from PySDM.backends.impl_thrust_rtc.nice_thrust import nice_thrust

from ..conf import trtc
from ..methods.thrust_rtc_backend_methods import ThrustRTCBackendMethods


class DisplacementMethods(ThrustRTCBackendMethods):
    def __init__(self):
        super().__init__()
        self.__calculate_displacement_body = trtc.For(
            (
                "dim",
                "n_sd",
                "displacement",
                "courant",
                "courant_length",
                "cell_origin",
                "position_in_cell",
                "n_substeps",
            ),
            "i",
            f"""
            // Arakawa-C grid
            auto _l_0 = cell_origin[i + 0];
            auto _l_1 = cell_origin[i + n_sd];
            auto _l = _l_0 + _l_1 * courant_length;
            auto _r_0 = cell_origin[i + 0] + 1 * (dim == 0);
            auto _r_1 = cell_origin[i + n_sd] + 1 * (dim == 1);
            auto _r = _r_0 + _r_1 * courant_length;
            auto omega = position_in_cell[i + n_sd * dim];
            auto c_r = courant[_r] / n_substeps;
            auto c_l = courant[_l] / n_substeps;
            displacement[i + n_sd * dim] = {
                self.formulae.particle_advection.displacement.c_inline(
                    c_l="c_l", c_r="c_r", omega="omega"
                )
            };
            """.replace(
                "real_type", self._get_c_type()
            ),
        )

        self.__flag_precipitated_body = trtc.For(
            (
                "idx",
                "n_sd",
                "n_dims",
                "healthy",
                "cell_origin",
                "position_in_cell",
                "volume",
                "n",
                "rainfall",
            ),
            "i",
            """
            auto origin = cell_origin[n_sd * (n_dims-1) + idx[i]];
            auto pic = position_in_cell[n_sd * (n_dims-1) + idx[i]];
            if (origin + pic < 0) {
                atomicAdd((real_type*) &rainfall[0], n[idx[i]] * volume[idx[i]]);
                idx[i] = n_sd;
                healthy[0] = 0;
            }
            """.replace(
                "real_type", self._get_c_type()
            ),
        )

    @nice_thrust(**NICE_THRUST_FLAGS)
    def calculate_displacement(
        self, *, dim, displacement, courant, cell_origin, position_in_cell, n_substeps
    ):
        dim = trtc.DVInt64(dim)
        n_sd = trtc.DVInt64(position_in_cell.shape[1])
        courant_length = trtc.DVInt64(courant.shape[0])
        self.__calculate_displacement_body.launch_n(
            displacement.shape[1],
            (
                dim,
                n_sd,
                displacement.data,
                courant.data,
                courant_length,
                cell_origin.data,
                position_in_cell.data,
                trtc.DVInt64(n_substeps),
            ),
        )

    @nice_thrust(**NICE_THRUST_FLAGS)
    def flag_precipitated(  # pylint: disable=unused-argument
        self,
        *,
        cell_origin,
        position_in_cell,
        volume,
        multiplicity,
        idx,
        length,
        healthy,
        precipitation_counting_level_index,
        displacement,
    ):
        if precipitation_counting_level_index != 0:
            raise NotImplementedError()
        n_sd = trtc.DVInt64(cell_origin.shape[1])
        n_dims = trtc.DVInt64(len(cell_origin.shape))
        rainfall = trtc.device_vector(self._get_c_type(), 1)
        trtc.Fill(rainfall, self._get_floating_point(0))
        self.__flag_precipitated_body.launch_n(
            length,
            [
                idx.data,
                n_sd,
                n_dims,
                healthy.data,
                cell_origin.data,
                position_in_cell.data,
                volume.data,
                multiplicity.data,
                rainfall,
            ],
        )
        return rainfall.to_host()[0]

    @staticmethod
    def flag_out_of_column(  # pylint: disable=unused-argument
        *, cell_origin, position_in_cell, idx, length, healthy, domain_top_level_index
    ):
        pass
