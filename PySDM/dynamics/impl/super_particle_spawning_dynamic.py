"""common code for dynamics that spawn new super-droplets"""

from PySDM.impl.particle_attributes import ParticleAttributes


class SuperParticleSpawningDynamic:  # pylint: disable=too-few-public-methods
    """base class for dynamics that spawn new super-droplets"""

    @staticmethod
    def check_extensive_attribute_keys(
        particulator_attributes: ParticleAttributes,
        spawned_attributes: dict,
    ):
        """checks if keys (and their order) in user-supplied `spawned_attributes` match
        attributes used in the `particulator_attributes`"""
        if tuple(particulator_attributes.get_extensive_attribute_keys()) != tuple(
            spawned_attributes.keys()
        ):
            raise ValueError(
                f"extensive attributes ({spawned_attributes.keys()})"
                " do not match those used in particulator"
                f" ({particulator_attributes.get_extensive_attribute_keys()})"
            )
