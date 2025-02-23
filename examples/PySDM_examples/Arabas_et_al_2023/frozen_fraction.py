class FrozenFraction:
    def __init__(self, *, volume, droplet_volume, total_particle_number, rho_w):
        self.volume = volume
        self.rho_w = rho_w
        self.droplet_volume = droplet_volume
        self.total_particle_number = total_particle_number

    def qi2ff(self, ice_mass_per_volume):
        ice_mass = ice_mass_per_volume * self.volume
        ice_number = ice_mass / (self.rho_w * self.droplet_volume)
        frozen_fraction = ice_number / self.total_particle_number
        return frozen_fraction

    def ff2qi(self, frozen_fraction):
        ice_number = frozen_fraction * self.total_particle_number
        ice_mass = ice_number * (self.rho_w * self.droplet_volume)
        ice_mass_per_volume = ice_mass / self.volume
        return ice_mass_per_volume
