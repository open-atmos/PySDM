"""
GPU implementation of backend methods for particle displacement (advection and sedimentation)
"""

from functools import cached_property

from PySDM.backends.impl_thrust_rtc.conf import NICE_THRUST_FLAGS
from PySDM.backends.impl_thrust_rtc.nice_thrust import nice_thrust

from ..conf import trtc
from ..methods.thrust_rtc_backend_methods import ThrustRTCBackendMethods


class DisplacementMethods(ThrustRTCBackendMethods):
    @cached_property
    def __calculate_displacement_body(self):
        return {
            n_dims: trtc.For(
                param_names=(
                    "dim",
                    "n_sd",
                    "displacement",
                    "courant",
                    "courant_shape_0",
                    "courant_shape_1",
                    "cell_origin",
                    "position_in_cell",
                    "n_substeps",
                ),
                name_iter="i",
                body=f"""
            // Arakawa-C grid
            auto _l = cell_origin[i];
            auto _r = cell_origin[i] + 1 * (dim == 0);

            if ({n_dims} > 1) {{
                auto _l_1 = cell_origin[i + n_sd];
                auto _r_1 = cell_origin[i + n_sd] + 1 * (dim == 1);

                _l += _l_1 * courant_shape_0;
                _r += _r_1 * courant_shape_0;
            }}

            if ({n_dims} > 2) {{
                auto _l_2 = cell_origin[i + 2 * n_sd];
                auto _r_2 = cell_origin[i + 2 * n_sd] + 1 * (dim == 2);

                _l += _l_2 * courant_shape_0 * courant_shape_1;
                _r += _r_2 * courant_shape_0 * courant_shape_1;
            }}

            displacement[i + n_sd * dim] = {
                self.formulae.particle_advection.displacement.c_inline(
                    position_in_cell="position_in_cell[i + n_sd * dim]",
                    c_l="courant[_l] / n_substeps",
                    c_r="courant[_r] / n_substeps"
                )
            };
            """.replace(
                    "real_type", self._get_c_type()
                ),
            )
            for n_dims in (1, 2, 3)
        }

    @cached_property
    def __flag_precipitated_body(self):
        return trtc.For(
            (
                "idx",
                "n_sd",
                "n_dims",
                "healthy",
                "cell_origin",
                "position_in_cell",
                "volume",
                "multiplicity",
                "rainfall",
            ),
            "i",
            """
            auto origin = cell_origin[n_sd * (n_dims-1) + idx[i]];
            auto pic = position_in_cell[n_sd * (n_dims-1) + idx[i]];
            if (origin + pic < 0) {
                atomicAdd((real_type*) &rainfall[0], multiplicity[idx[i]] * volume[idx[i]]);
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
        n_dim = len(courant.shape)
        n_sd = position_in_cell.shape[1]
        self.__calculate_displacement_body[n_dim].launch_n(
            n=n_sd,
            args=(
                trtc.DVInt64(dim),
                trtc.DVInt64(n_sd),
                displacement.data,
                courant.data,
                trtc.DVInt64(courant.shape[0]),
                trtc.DVInt64(courant.shape[1] if n_dim > 2 else -1),
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
            (
                idx.data,
                n_sd,
                n_dims,
                healthy.data,
                cell_origin.data,
                position_in_cell.data,
                volume.data,
                multiplicity.data,
                rainfall,
            ),
        )
        return rainfall.to_host()[0]

    @staticmethod
    def flag_out_of_column(  # pylint: disable=unused-argument
        *, cell_origin, position_in_cell, idx, length, healthy, domain_top_level_index
    ):
        pass
