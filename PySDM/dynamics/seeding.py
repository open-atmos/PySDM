""" particle injection handling, requires initalising a simulation with
enough particles flagged with NaN multiplicity (translated to zeros
at multiplicity discretisation """

from PySDM.dynamics.impl import register_dynamic


@register_dynamic()
class Seeding:
    def __init__(
        self,
        *,
        super_droplet_injection_rate: callable,
        seeded_particle_extensive_attributes: dict,
        seeded_particle_multiplicity: int,
    ):
        self.particulator = None
        self.super_droplet_injection_rate = super_droplet_injection_rate
        self.seeded_particle_extensive_attributes = seeded_particle_extensive_attributes
        self.seeded_particle_multiplicity = seeded_particle_multiplicity

    def register(self, builder):
        self.particulator = builder.particulator

    def __call__(self):
        if self.particulator.n_steps == 0:
            if tuple(
                self.particulator.attributes.get_extensive_attribute_keys()
            ) != tuple(self.seeded_particle_extensive_attributes.keys()):
                raise ValueError(
                    f"extensive attributes ({self.seeded_particle_extensive_attributes.keys()})"
                    " do not match those used in particulator"
                    f" ({self.particulator.attributes.get_extensive_attribute_keys()})"
                )

        time = self.particulator.n_steps * self.particulator.dt
        self.particulator.seeding(
            number_of_super_particles_to_inject=self.super_droplet_injection_rate(time),
            seeded_particle_multiplicity=self.seeded_particle_multiplicity,
            seeded_particle_extensive_attributes=self.seeded_particle_extensive_attributes,
        )
