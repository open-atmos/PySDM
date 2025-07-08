"""
droplet freezing using either singular or
time-dependent formulation for immersion freezing
and homogeneous freezing and thaw
"""

from typing import Optional
from PySDM.dynamics.impl import register_dynamic


@register_dynamic()
class Freezing:  # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        *,
        homogeneous_freezing: Optional[str] = None,
        immersion_freezing: Optional[str] = None,
        thaw: Optional[str] = None,
    ):
        assert (
            homogeneous_freezing or immersion_freezing or thaw
        ), "please enable one or more modes of operation"
        assert immersion_freezing is None or immersion_freezing in (
            "time-dependent",
            "singular",
        )
        assert homogeneous_freezing is None or homogeneous_freezing in (
            "time-dependent",
            "threshold",
        )
        assert thaw is None or thaw == "instantaneous"

        self.homogeneous_freezing = homogeneous_freezing
        self.immersion_freezing = immersion_freezing
        self.thaw = thaw
        self.enable = True
        self.rand = None
        self.rng = None
        self.particulator = None

    def register(self, builder):
        self.particulator = builder.particulator

        assert (
            self.particulator.formulae.particle_shape_and_density.supports_mixed_phase()
        )

        builder.request_attribute("signed water mass")
        if self.immersion_freezing == "singular":
            builder.request_attribute("freezing temperature")

        if self.immersion_freezing == "time-dependent":
            assert (
                self.particulator.formulae.heterogeneous_ice_nucleation_rate.__name__
                != "Null"
            )
            builder.request_attribute("immersed surface area")

        if self.homogeneous_freezing == "time-dependent":
            assert (
                self.particulator.formulae.homogeneous_ice_nucleation_rate.__name__
                != "Null"
            )
            builder.request_attribute("volume")

        if (
            self.homogeneous_freezing == "time-dependent"
            or self.immersion_freezing == "time-dependent"
        ):
            self.rand = self.particulator.Storage.empty(
                self.particulator.n_sd, dtype=float
            )
            self.rng = self.particulator.Random(
                self.particulator.n_sd, self.particulator.formulae.seed
            )

    def __call__(self):
        if "Coalescence" in self.particulator.dynamics:
            # TODO #594
            raise NotImplementedError(
                "handling T_fz during collisions not implemented yet"
            )

        if not self.enable:
            return

        if self.immersion_freezing == "singular":
            self.particulator.immersion_freezing_singular()
        elif self.immersion_freezing == "time-dependent":
            self.rand.urand(self.rng)
            self.particulator.immersion_freezing_time_dependent(
                rand=self.rand,
            )

        if self.homogeneous_freezing == "threshold":
            self.particulator.homogeneous_freezing_threshold()
        elif self.homogeneous_freezing == "time-dependent":
            self.rand.urand(self.rng)
            self.particulator.homogeneous_freezing_time_dependent(
                rand=self.rand,
            )

        if self.thaw == "instantaneous":
            self.particulator.thaw_instantaneous()
