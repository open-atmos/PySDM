"""
CPU implementation of backend methods for removal due to sedimentation
in a 0D environment
"""

from functools import cached_property

import numba

from PySDM.backends.impl_common.backend_methods import BackendMethods


class SedimentationRemoval0DMethods(BackendMethods):

    @cached_property
    def _sedimentation_removal_deterministic_body(self):

        @numba.njit(**self.default_jit_flags)
        def body(relative_fall_velocity, multiplicity, length_scale, timestep):
            n_sd = len(relative_fall_velocity)
            for i in numba.prange(n_sd):  # pylint: disable=not-an-iterable
                multiplicity[i] -= (
                    multiplicity[i]
                    * relative_fall_velocity[i]
                    * timestep
                    / length_scale
                )

        return body

    @cached_property
    def _sedimentation_removal_stochastic_body(self):

        prob_zero_events = self.formulae.trivia.poissonian_avoidance_function

        @numba.njit(**self.default_jit_flags)
        def body(relative_fall_velocity, multiplicity, length_scale, timestep):
            n_sd = len(relative_fall_velocity)
            for i in numba.prange(n_sd):  # pylint: disable=not-an-iterable
                removal_rate = relative_fall_velocity[i] / length_scale
                survive_prob = prob_zero_events(r=removal_rate, dt=timestep)
                multiplicity[i] *= survive_prob

        return body

    def sedimentation_removal_deterministic(
        self, *, relative_fall_velocity, multiplicity, length_scale, timestep
    ):
        self._sedimentation_removal_deterministic_body(
            relative_fall_velocity, multiplicity, length_scale, timestep
        )

    def sedimentation_removal_stochastic(
        self, *, relative_fall_velocity, multiplicity, length_scale, timestep
    ):
        self._sedimentation_removal_stochastic_body(
            relative_fall_velocity, multiplicity, length_scale, timestep
        )
