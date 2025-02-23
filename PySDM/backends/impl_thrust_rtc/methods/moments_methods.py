"""
GPU implementation of moment calculation backend methods
"""

from functools import cached_property

from PySDM.backends.impl_thrust_rtc.conf import NICE_THRUST_FLAGS
from PySDM.backends.impl_thrust_rtc.nice_thrust import nice_thrust

from ..conf import trtc
from ..methods.thrust_rtc_backend_methods import ThrustRTCBackendMethods

# https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
DOUBLE_ATOMIC_ADD_FOR_COMPUTE_LT_60 = """
struct Commons {
    static __device__ double atomicAdd(double* address, double val)
    {
        unsigned long long int* address_as_ull =
                                  (unsigned long long int*)address;
        unsigned long long int old = *address_as_ull, assumed;

        do {
            assumed = old;
            old = atomicCAS(address_as_ull, assumed,
                            __double_as_longlong(val +
                                   __longlong_as_double(assumed)));

        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
        } while (assumed != old);

        return __longlong_as_double(old);
    }
};
double (*atomicAdd)(double*, double) = &Commons::atomicAdd;
"""


class MomentsMethods(ThrustRTCBackendMethods):
    def __init__(self, double_precision):
        ThrustRTCBackendMethods.__init__(self)

        if trtc.Get_PTX_Arch() < 60 and double_precision:
            self.commons = DOUBLE_ATOMIC_ADD_FOR_COMPUTE_LT_60
        else:
            self.commons = ""

    @cached_property
    def __moments_body_0(self):
        return trtc.For(
            (
                "idx",
                "min_x",
                "attr_data",
                "x_attr",
                "max_x",
                "moment_0",
                "cell_id",
                "multiplicity",
                "n_ranks",
                "moments",
                "ranks",
                "n_sd",
                "n_cell",
            ),
            "fake_i",
            self.commons
            + """
            auto i = idx[fake_i];
            if (min_x <= x_attr[i] && x_attr[i] < max_x) {
                atomicAdd((real_type*)&moment_0[cell_id[i]], (real_type)(multiplicity[i]));
                for (auto k = 0; k < n_ranks; k+=1) {
                    auto value = multiplicity[i] * pow(
                        (real_type)(attr_data[i]),
                        (real_type)(ranks[k])
                    );
                    atomicAdd((real_type*) &moments[n_cell * k + cell_id[i]], value);
               }
            }
        """.replace(
                "real_type", self._get_c_type()
            ),
        )

    @cached_property
    def __moments_body_1(self):
        return trtc.For(
            ("n_ranks", "moments", "moment_0", "n_cell"),
            "c_id",
            """
            for (auto k = 0; k < n_ranks; k+=1) {
                if (moment_0[c_id] == 0) {
                    moments[n_cell * k  + c_id] = 0;
                }
                else {
                    moments[n_cell * k + c_id] = moments[n_cell * k + c_id] / moment_0[c_id];
                }
            }
        """,
        )

    @cached_property
    def __spectrum_moments_body_0(self):
        return trtc.For(
            (
                "idx",
                "attr_data",
                "x_attr",
                "moment_0",
                "cell_id",
                "multiplicity",
                "x_bins",
                "n_bins",
                "moments",
                "rank",
                "n_sd",
                "n_cell",
            ),
            "fake_i",
            self.commons
            + """
            auto i = idx[fake_i];
            for (auto k = 0; k < n_bins; k+=1) {
                if (x_bins[k] <= x_attr[i] and x_attr[i] < x_bins[k + 1]) {
                    atomicAdd(
                        (real_type*)&moment_0[n_cell * k + cell_id[i]],
                        (real_type)(multiplicity[i])
                    );
                    auto val = multiplicity[i] * pow((real_type)(attr_data[i]), (real_type)(rank));
                    atomicAdd((real_type*) &moments[n_cell * k + cell_id[i]], val);
                    break;
                }
            }
        """.replace(
                "real_type", self._get_c_type()
            ),
        )

    @cached_property
    def __spectrum_moments_body_1(self):
        return trtc.For(
            ("n_bins", "moments", "moment_0", "n_cell"),
            "i",
            """
            for (auto k = 0; k < n_bins; k+=1) {
                if (moment_0[n_cell * k + i] == 0) {
                    moments[n_cell * k  + i] = 0;
                }
                else {
                    moments[n_cell * k + i] = moments[n_cell * k + i] / moment_0[n_cell * k + i];
                }
            }
        """,
        )

    @staticmethod
    def ensure_floating_point(data):
        data_type = data.data.name_elem_cls()
        if data_type not in ("float", "double"):
            raise TypeError(f"data type {data_type} not recognised as floating point")

    # TODO #684
    # pylint: disable=unused-argument,too-many-locals
    @nice_thrust(**NICE_THRUST_FLAGS)
    def moments(
        self,
        *,
        moment_0,
        moments,
        multiplicity,
        attr_data,
        cell_id,
        idx,
        length,
        ranks,
        min_x,
        max_x,
        x_attr,
        weighting_attribute,
        weighting_rank,
        skip_division_by_m0,
    ):
        if weighting_rank != 0:
            raise NotImplementedError()

        self.ensure_floating_point(moment_0)
        self.ensure_floating_point(moments)

        n_cell = trtc.DVInt64(moments.shape[1])
        n_sd = trtc.DVInt64(moments.shape[0])
        n_ranks = trtc.DVInt64(ranks.shape[0])

        moments[:] = 0
        moment_0[:] = 0

        self.__moments_body_0.launch_n(
            length,
            (
                idx.data,
                self._get_floating_point(min_x),
                attr_data.data,
                x_attr.data,
                self._get_floating_point(max_x),
                moment_0.data,
                cell_id.data,
                multiplicity.data,
                n_ranks,
                moments.data,
                ranks.data,
                n_sd,
                n_cell,
            ),
        )

        if not skip_division_by_m0:
            self.__moments_body_1.launch_n(
                moment_0.shape[0], (n_ranks, moments.data, moment_0.data, n_cell)
            )

    # TODO #684
    # pylint: disable=unused-argument,too-many-locals
    @nice_thrust(**NICE_THRUST_FLAGS)
    def spectrum_moments(
        self,
        *,
        moment_0,
        moments,
        multiplicity,
        attr_data,
        cell_id,
        idx,
        length,
        rank,
        x_bins,
        x_attr,
        weighting_attribute,
        weighting_rank,
    ):
        assert moments.shape[0] == x_bins.shape[0] - 1
        assert moment_0.shape == moments.shape
        if weighting_rank != 0:
            raise NotImplementedError()

        self.ensure_floating_point(moment_0)
        self.ensure_floating_point(moments)

        n_cell = trtc.DVInt64(moments.shape[1])
        n_sd = trtc.DVInt64(moments.shape[0])
        d_rank = trtc.DVInt64(rank)
        n_bins = trtc.DVInt64(len(x_bins) - 1)

        moments[:] = 0
        moment_0[:] = 0

        self.__spectrum_moments_body_0.launch_n(
            length,
            (
                idx.data,
                attr_data.data,
                x_attr.data,
                moment_0.data,
                cell_id.data,
                multiplicity.data,
                x_bins.data,
                n_bins,
                moments.data,
                d_rank,
                n_sd,
                n_cell,
            ),
        )

        self.__spectrum_moments_body_1.launch_n(
            moment_0.shape[1], (n_bins, moments.data, moment_0.data, n_cell)
        )
