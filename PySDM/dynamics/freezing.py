"""
droplet freezing using either singular or
time-dependent formulation for immersion freezing
and homogeneous freezing and thaw
"""

from PySDM.dynamics.impl import register_dynamic


@register_dynamic()
class Freezing:  # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        *,
        singular=True,
        homogeneous_freezing=False,
        immersion_freezing=True,
        thaw=False,
    ):
        self.singular = singular
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
        if self.singular:
            builder.request_attribute("freezing temperature")

        if not self.singular and self.immersion_freezing:
            assert (
                self.particulator.formulae.heterogeneous_ice_nucleation_rate.__name__
                != "Null"
            )
            builder.request_attribute("immersed surface area")

        if self.homogeneous_freezing:
            assert (
                self.particulator.formulae.homogeneous_ice_nucleation_rate.__name__
                != "Null"
            )
            builder.request_attribute("volume")

        if self.homogeneous_freezing or not self.singular:
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

        if self.immersion_freezing:
            if self.singular:
                self.particulator.immersion_freezing_singular(thaw=self.thaw)
            else:
                self.rand.urand(self.rng)
                self.particulator.immersion_freezing_time_dependent(
                    rand=self.rand,
                    thaw=self.thaw,
                )

        if self.homogeneous_freezing:
            self.rand.urand(self.rng)
            self.particulator.homogeneous_freezing_time_dependent(
                rand=self.rand,
                thaw=self.thaw,
            )
