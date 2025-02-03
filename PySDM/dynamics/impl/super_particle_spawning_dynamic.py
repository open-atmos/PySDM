from PySDM.impl.particle_attributes import ParticleAttributes


class SuperParticleSpawningDynamic:
    @staticmethod
    def check_extensive_attribute_keys(
        particulator_attributes: ParticleAttributes,
        spawned_attributes: dict,
    ):
        if tuple(particulator_attributes.get_extensive_attribute_keys()) != tuple(
            spawned_attributes.keys()
        ):
            raise ValueError(
                f"extensive attributes ({spawned_attributes.keys()})"
                " do not match those used in particulator"
                f" ({particulator_attributes.get_extensive_attribute_keys()})"
            )
